#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"


#define NTHREADS 1024


void __hip_check_error(hipError_t err, const char *file, int line){
	if(err != hipSuccess){
        fprintf(stderr, "HIP error (%s:%d): %s\n", file, line, hipGetErrorString(err));
        exit(1);
    }
}


#define HIP_CHECK_ERROR(X)({\
	__hip_check_error((X), __FILE__, __LINE__);\
})



#define MALLOC_CHECK_ERROR(X)({\
    if ((X) == 0){\
        fprintf(stderr, "Malloc error (%s:%d)\n", __FILE__, __LINE__);\
        exit(1);\
    }\
})



__global__ void vector_max(int *v, int *max, int n){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int lane_id = threadIdx.x % warpSize;
    unsigned int warp_id = threadIdx.x / warpSize;
    __shared__ int block_max[warpSize];
    if (idx < n && lane_id == 0) block_max[warp_id] = v[idx];
    __syncthreads();
   // TODO: continue here
}


// Returns the maximum value in the vector v of size n.
int vector_max_driver(int *v, int n){
    int *dev_v, *dev_max, max;
    HIP_CHECK_ERROR(hipMalloc(&dev_v, sizeof(int) * n));
    HIP_CHECK_ERROR(hipMalloc(&dev_max, sizeof(int)));
    HIP_CHECK_ERROR(hipMemcpy(dev_v, v, sizeof(int) * n, hipMemcpyHostToDevice));
    // set the max as the first element of the array
    HIP_CHECK_ERROR(hipMemcpy(dev_max, v, sizeof(int), hipMemcpyHostToDevice));
    // setup kernel configuration
    unsigned int nBlocks = (n + NTHREADS - 1) / NTHREADS;
    hipEvent_t start, stop;
    HIP_CHECK_ERROR(hipEventCreate(&start));
    HIP_CHECK_ERROR(hipEventCreate(&stop));
    HIP_CHECK_ERROR(hipEventRecord(start));
    vector_max<<<nBlocks, NTHREADS>>>(dev_v, dev_max, n);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipEventRecord(stop));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    float elapsed;
    HIP_CHECK_ERROR(hipEventElapsedTime(&elapsed, start, stop));
    printf("'vector_max' kernel execution time (ms): %.3lf\n", elapsed);
    HIP_CHECK_ERROR(hipMemcpy(&max, dev_max, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    HIP_CHECK_ERROR(hipFree(dev_v));
    HIP_CHECK_ERROR(hipFree(dev_max));
    HIP_CHECK_ERROR(hipEventDestroy(start));
    HIP_CHECK_ERROR(hipEventDestroy(stop));
    return max;
}



void test_correctness(void){
    int n = 10;
    int v[] = {1, 2, 3, 340, 4, 5, 6, 7, 8, 9};
    int max_check = 340;
    int result = vector_max_driver(v, n);
    if(max_check != result){
        fprintf(stderr, "Max is not correct.\n");
        exit(1);
    }
    printf("Correctness test: all good.\n");
}



void test_performance(void){
    int n = 1e9;
    int *v = (int *) malloc(sizeof(int) * n);
    MALLOC_CHECK_ERROR(v);
    int max_check = -1;
    for(int i = 0; i < n; i++){
        v[i] = rand() % 3054;
    }
    ptimer_t cpu_timer;
    timer_start(&cpu_timer);
    for(int i = 0; i < n; i++){
        if(v[i] > max_check) max_check = v[i];
    }
    timer_stop(&cpu_timer);
    printf("Execution time on CPU (ms): %.3lf\n", timer_elapsed(cpu_timer));
    ptimer_t gpu_timer;
    timer_start(&gpu_timer);
    int result = vector_max_driver(v, n);
    free(v);
    timer_stop(&gpu_timer);
    printf("Execution time on GPU (ms): %.3lf\n", timer_elapsed(gpu_timer));
    if(max_check != result){
        fprintf(stderr, "Performance test: max is not correct!\n");
        exit(1);
    }else{
        printf("Peformance test: all good.\n");
    }    
}



int main(void){
    test_correctness();
    test_performance();
    return 0;
}
