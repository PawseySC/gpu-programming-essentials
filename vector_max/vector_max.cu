#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"


#define NTHREADS 1024



#define CUDA_CHECK_ERROR(X)({\
    if((X) != cudaSuccess){\
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString((X)));\
        exit(1);\
    }\
})



#define MALLOC_CHECK_ERROR(X)({\
    if ((X) == 0){\
        fprintf(stderr, "Malloc error (%s:%d): %i\n", __FILE__, __LINE__, (X));\
        exit(1);\
    }\
})



__global__ void vector_max(int *v, int *max, int n){
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int warp_id = threadIdx.x / 32;
    __shared__ int block_max[32];
    if (idx < n && lane_id == 0) block_max[warp_id] = v[idx];
    __syncthreads();
   // TODO: continue here
}


// Returns the maximum value in the vector v of size n.
int vector_max_driver(int *v, int n){
    int *dev_v, *dev_max, max;
    CUDA_CHECK_ERROR(cudaMalloc(&dev_v, sizeof(int) * n));
    CUDA_CHECK_ERROR(cudaMalloc(&dev_max, sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_v, v, sizeof(int) * n, cudaMemcpyHostToDevice));
    // set the max as the first element of the array
    CUDA_CHECK_ERROR(cudaMemcpy(dev_max, v, sizeof(int), cudaMemcpyHostToDevice));
    // setup kernel configuration
    unsigned int nBlocks = (n + NTHREADS - 1) / NTHREADS;
    ptimer_t kernel_timer;
    timer_start(&kernel_timer);
    vector_max<<<nBlocks, NTHREADS>>>(dev_v, dev_max, n);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    timer_stop(&kernel_timer);
    printf("'vector_max' kernel execution time (ms): %.3lf\n", timer_elapsed(kernel_timer));
    CUDA_CHECK_ERROR(cudaMemcpy(&max, dev_max, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaFree(dev_v));
    CUDA_CHECK_ERROR(cudaFree(dev_max));
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