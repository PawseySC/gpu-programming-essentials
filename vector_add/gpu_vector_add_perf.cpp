#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"
#include "../common/array.h"

#define NTHREADS 1024


void __cuda_check_error(hipError_t err, const char *file, int line){
	if(err != hipSuccess){
        fprintf(stderr, "CUDA error (%s:%d): %s\n", file, line, hipGetErrorString(err));
        exit(1);
    }
}


#define CUDA_CHECK_ERROR(X)({\
	__cuda_check_error((X), __FILE__, __LINE__);\
})



#define MALLOC_CHECK_ERROR(X)({\
    if ((X) == 0){\
        fprintf(stderr, "Malloc error (%s:%d): %i\n", __FILE__, __LINE__, (X));\
        exit(1);\
    }\
})



// Returns True if |a - b| <= eps
inline char compare_float(float a, float b){
    const float eps = 1e-7f;
    if (a  > b) return a - b <= eps;
    else return b - a <= eps;
}



// Initialise the vector v of n elements to random values
void init_vec(float *v, int n){
    for(int i = 0; i < n; i++){
        v[i] = rand() % 100 * 0.3234f;
    }
}



// adds two vectors of length n.
void cpu_vector_add(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}



// kernel to perform vector addition
__global__ void vector_add(float *A, float *B, float *C, unsigned int n){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        C[i] = A[i] + B[i];
}



void vector_add_driver(float *A, float *B, float *C, unsigned int n){
    float *dev_A, *dev_B, *dev_C;
    CUDA_CHECK_ERROR(hipMalloc(&dev_A, sizeof(float) * n));
    CUDA_CHECK_ERROR(hipMalloc(&dev_B, sizeof(float) * n));
    CUDA_CHECK_ERROR(hipMalloc(&dev_C, sizeof(float) * n));
    CUDA_CHECK_ERROR(hipMemcpy(dev_A, A, sizeof(float) * n, hipMemcpyHostToDevice));
    CUDA_CHECK_ERROR(hipMemcpy(dev_B, B, sizeof(float) * n, hipMemcpyHostToDevice));
    const unsigned int nblocks = (n + NTHREADS - 1) / NTHREADS;
    hipEvent_t start, stop;
    CUDA_CHECK_ERROR(hipEventCreate(&start));
    CUDA_CHECK_ERROR(hipEventCreate(&stop));
    CUDA_CHECK_ERROR(hipEventRecord(start));
    hipLaunchKernelGGL(vector_add, nblocks, NTHREADS, 0, 0, dev_A, dev_B, dev_C, n);
    CUDA_CHECK_ERROR(hipGetLastError());
    CUDA_CHECK_ERROR(hipEventRecord(stop));
    CUDA_CHECK_ERROR(hipDeviceSynchronize());
    float elapsed;
    CUDA_CHECK_ERROR(hipEventElapsedTime(&elapsed, start, stop));
    printf("'vector_sum' kernel execution time (ms): %.3lf\n", elapsed);
    CUDA_CHECK_ERROR(hipMemcpy(C, dev_C, sizeof(float) * n, hipMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(hipDeviceSynchronize());
    CUDA_CHECK_ERROR(hipFree(dev_A));
    CUDA_CHECK_ERROR(hipFree(dev_B));
    CUDA_CHECK_ERROR(hipFree(dev_C));
    CUDA_CHECK_ERROR(hipEventDestroy(start));
    CUDA_CHECK_ERROR(hipEventDestroy(stop));
}



void test_correctness(void){
    const unsigned int n = 100;
    float *A = (float*) malloc(n * sizeof(float));
    float *B = (float*) malloc(n * sizeof(float));
    float *C = (float*) malloc(n * sizeof(float));
    MALLOC_CHECK_ERROR(A && B && C);
    
    // initialise A and B
    for(int i = 0; i < n; i++){
        A[i] = i;
        B[i] = n - i;
    }
    vector_add_driver(A,B, C, n);
    // test correctness of result
    char test_passed = 1;
    for(unsigned int i = 0; i < n; i++){
        if(!compare_float(C[i], n)){
            test_passed = 0;
            break;
        }
    }
    free(A);
    free(B);
    free(C);
    if(test_passed){
        printf("Correctness: all good.\n");
    }else{
        fprintf(stderr, "Correctness: failed!\n");
    }
}



void test_performance(unsigned int n){
    float *A = (float*) malloc(n * sizeof(float));
    float *B = (float*) malloc(n * sizeof(float));
    float *C_gpu = (float*) malloc(n * sizeof(float));
    float *C_cpu = (float*) malloc(n * sizeof(float));
    MALLOC_CHECK_ERROR(A && B && C_gpu && C_cpu);
    init_vec(A, n);
    init_vec(B, n);
    // GPU computation
    ptimer_t timer_gpu;
    timer_start(&timer_gpu);
    vector_add_driver(A, B, C_gpu, n);
    timer_stop(&timer_gpu);
    printf("'vector_add_diver' execution time (ms): %.3lf\n", timer_elapsed(timer_gpu));
   
    // CPU computation
    ptimer_t timer_cpu;
    timer_start(&timer_cpu);
    cpu_vector_add(A, B, C_cpu, n);
    timer_stop(&timer_cpu);
    printf("'cpu_vector_add' execution time (ms): %.3lf\n", timer_elapsed(timer_cpu));
    

    printf("Array A: ");
    print_array_terse(A, n, 3);
    printf("Array B: ");
    print_array_terse(B, n, 3);
    printf("Array C_gpu: ");
    print_array_terse(C_gpu, n, 3);
    printf("Array C_cpu: ");
    print_array_terse(C_cpu, n, 3);
    
    char test_passed = 1;
    for(unsigned int i = 0; i < n; i++){
        if(!compare_float(C_cpu[i], C_gpu[i])){
            test_passed = 0;
            break;
        }
    }
    free(A);
    free(B);
    free(C_gpu);
    free(C_cpu);
    if(test_passed)
        printf("Performance: all good.\n");
    else
        fprintf(stderr, "Performance: wrong result!\n");
}


int main(int argc, char **argv){
    unsigned int n = 1e9;
    if(argc >= 2){
        n = (unsigned int) atoi(argv[1]);
    }else{
        printf("Using default n = 100 for performance testing.\nTo change behaviour, %s [n]\n\n", argv[0]);
    }
    // The following call is done to initialise CUDA here and not to include CUDA initialisation in subsequent calls, which we are timing.
    CUDA_CHECK_ERROR(hipSetDevice(0));
    test_correctness();
    test_performance(n);
    return 0;
}
