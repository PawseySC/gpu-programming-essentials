#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void __hip_check_error(hipError_t err, const char *file, int line){
	if(err != hipSuccess){
        fprintf(stderr, "HIP error (%s:%d): %s\n", file, line, hipGetErrorString(err));
        exit(1);
    }
}


#define HIP_CHECK_ERROR(X)({\
	__hip_check_error((X), __FILE__, __LINE__);\
})

#define NTHREADS 1024 



__global__ void vector_reduction_kernel(unsigned char *values, unsigned int nitems, unsigned long long* result){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int partial_sums[NTHREADS];
    unsigned int warpId = threadIdx.x / warpSize;
    unsigned int laneId = threadIdx.x % warpSize; 
    if(idx < nitems){ 
        partial_sums[threadIdx.x] = values[idx];  
    }else{
        partial_sums[threadIdx.x] = 0;  
    }
    __syncthreads();
 
    // step 1
    for(unsigned int i = warpSize/2; i >= 1; i >>= 1){
        if(laneId < i){
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + i];
        }
    }
    __syncthreads();
    // step 2
    if(warpId == 0){
        for(unsigned int i = warpSize/2; i >= 1; i >>= 1){
            if(laneId < i){
                partial_sums[warpSize*laneId] += partial_sums[warpSize*(laneId + i)];
            }
        }
        if(laneId == 0) atomicAdd(result, partial_sums[0]);
    }  
}



int main(int argc, char **argv){
    
    unsigned int nitems = 1e9; 
    unsigned char *values = (unsigned char*) malloc(sizeof(unsigned int) * nitems);
    if(!values){
        fprintf(stderr, "Error while allocating memory\n");
        return EXIT_FAILURE;
    }
    // Initialise the vector of n elements to random values
    unsigned long long correct_result = 0;
    for(int i = 0; i < nitems; i++){
        values[i] = (i + 1) % 128;
        correct_result += values[i];
    }
    unsigned long long sum = 0ull;
    unsigned long long *dev_sum;
    unsigned char *dev_values;
    HIP_CHECK_ERROR(hipMalloc(&dev_values, sizeof(unsigned char) * nitems));
    HIP_CHECK_ERROR(hipMalloc(&dev_sum, sizeof(unsigned long long)));
    HIP_CHECK_ERROR(hipMemset(dev_sum, 0, sizeof(unsigned long long)));
    HIP_CHECK_ERROR(hipMemcpy(dev_values, values, sizeof(unsigned char) * nitems, hipMemcpyHostToDevice));
    unsigned int nblocks = (nitems + NTHREADS - 1) / NTHREADS;
    printf("Number of hip blocks: %u\n", nblocks);
    hipEvent_t start, stop;
    HIP_CHECK_ERROR(hipEventCreate(&start));
    HIP_CHECK_ERROR(hipEventCreate(&stop));
    HIP_CHECK_ERROR(hipEventRecord(start)); 
    hipLaunchKernelGGL(vector_reduction_kernel, nblocks, NTHREADS, 0, 0, dev_values, nitems, dev_sum);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipEventRecord(stop)); 
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    HIP_CHECK_ERROR(hipMemcpy(&sum, dev_sum, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    float time_spent;
    HIP_CHECK_ERROR(hipEventElapsedTime(&time_spent, start, stop));
    printf("Result: %llu - Time elapsed: %f\n", sum, time_spent/1000.0f);
    if(correct_result != sum) {
        fprintf(stderr, "Error: sum is not correct, should be %llu\n", correct_result);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
