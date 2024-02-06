#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define HIP_CHECK_ERROR(X)({\
    if((X) != hipSuccess){\
        fprintf(stderr, "ERROR %d (%s:%d): %s\n", (X), __FILE__, __LINE__, hipGetErrorString((X)));\
        exit(1);\
    }\
})

#define NTHREADS 1024
#define NWARPS 16


__global__ void vector_sum(unsigned char *values, unsigned int nitems, unsigned long long* result){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ unsigned int partial_sums[NWARPS];
    unsigned int warpId = threadIdx.x / warpSize;
    unsigned int laneId = threadIdx.x % warpSize; 
    unsigned int gridSize = gridDim.x * blockDim.x;
    unsigned int nloops = (nitems + gridSize  - 1) / gridSize;
    unsigned int myvalue = 0;
    for(unsigned int l = 0; l < nloops; l++, idx+=gridSize){
        if(idx < nitems) myvalue += values[idx]; 
    }
 
    for(unsigned int i = warpSize/2; i >= 1; i /= 2){
        unsigned int up = __shfl_down(myvalue, i); 
        if(laneId < i){
            myvalue += up; 
        }
    }
    if(laneId == 0 && warpId > 0) partial_sums[warpId] = myvalue;

    __syncthreads();
    // step 2
    if(warpId == 0){
        if(laneId > 0 && laneId < NWARPS) myvalue = partial_sums[laneId];
        for(unsigned int i = NWARPS/2; i >= 1; i >>= 1){
            unsigned int up = __shfl_down(myvalue, i, NWARPS); 
            if(laneId < i){
               myvalue += up; 
            }
        }
        if(laneId == 0) atomicAdd(result, myvalue);
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
    struct hipDeviceProp_t props;
    HIP_CHECK_ERROR(hipGetDeviceProperties(&props, 0));
    unsigned int nblocks = props.multiProcessorCount * 2;
    printf("Number of hip blocks: %u\n", nblocks);
    hipEvent_t start, stop;
    HIP_CHECK_ERROR(hipEventCreate(&start));
    HIP_CHECK_ERROR(hipEventCreate(&stop));
    HIP_CHECK_ERROR(hipEventRecord(start)); 
    hipLaunchKernelGGL(vector_sum, nblocks, NTHREADS, 0, 0, dev_values, nitems, dev_sum);
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
