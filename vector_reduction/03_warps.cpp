#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void __cuda_check_error(hipError_t err, const char *file, int line){
	if(err != hipSuccess){
        fprintf(stderr, "CUDA error (%s:%d): %s\n", file, line, hipGetErrorString(err));
        exit(1);
    }
}


#define CUDA_CHECK_ERROR(X)({\
	__cuda_check_error((X), __FILE__, __LINE__);\
})

#define NTHREADS 1024 
#define WARPSIZE 32


__global__ void vector_reduction_kernel(unsigned char *values, unsigned int nitems, unsigned long long* result){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int partial_sums[WARPSIZE];
    unsigned int warpId = threadIdx.x / warpSize;
    unsigned int laneId = threadIdx.x % warpSize; 
    if(laneId == 0) partial_sums[warpId] = 0;
    __syncthreads();
    if(idx < nitems){ 
        atomicAdd(partial_sums + warpId, values[idx]);  
    }
    __syncthreads();
    if(laneId == 0 && warpId > 0) atomicAdd(partial_sums, partial_sums[warpId]);
    __syncthreads();
    if(threadIdx.x == 0) atomicAdd(result, partial_sums[0]);
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
    CUDA_CHECK_ERROR(hipMalloc(&dev_values, sizeof(unsigned char) * nitems));
    CUDA_CHECK_ERROR(hipMalloc(&dev_sum, sizeof(unsigned long long)));
    CUDA_CHECK_ERROR(hipMemset(dev_sum, 0, sizeof(unsigned long long)));
    CUDA_CHECK_ERROR(hipMemcpy(dev_values, values, sizeof(unsigned char) * nitems, hipMemcpyHostToDevice));
    unsigned int nblocks = (nitems + NTHREADS - 1) / NTHREADS;
    printf("Number of cuda blocks: %u\n", nblocks);
    hipEvent_t start, stop;
    CUDA_CHECK_ERROR(hipEventCreate(&start));
    CUDA_CHECK_ERROR(hipEventCreate(&stop));
    CUDA_CHECK_ERROR(hipEventRecord(start)); 
    hipLaunchKernelGGL(vector_reduction_kernel, nblocks, NTHREADS, 0, 0, dev_values, nitems, dev_sum);
    CUDA_CHECK_ERROR(hipGetLastError());
    CUDA_CHECK_ERROR(hipEventRecord(stop)); 
    CUDA_CHECK_ERROR(hipDeviceSynchronize());
    CUDA_CHECK_ERROR(hipMemcpy(&sum, dev_sum, sizeof(unsigned long long), hipMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(hipDeviceSynchronize());
    float time_spent;
    CUDA_CHECK_ERROR(hipEventElapsedTime(&time_spent, start, stop));
    printf("Result: %llu - Time elapsed: %f\n", sum, time_spent/1000.0f);
    if(correct_result != sum) {
        fprintf(stderr, "Error: sum is not correct, should be %llu\n", correct_result);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}
