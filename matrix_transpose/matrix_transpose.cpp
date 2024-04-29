#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include "../common/timer.h"


#define NTHREADS 1024

#define HIP_CHECK_ERROR(X)({\
    if((X) != hipSuccess){\
        fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString((X)));\
        exit(1);\
    }\
})


#define MALLOC_CHECK_ERROR(X)({\
    if ((X) == 0){\
        fprintf(stderr, "Malloc error (%s:%d).\n", __FILE__, __LINE__);\
        exit(1);\
    }\
})


// The following is the CPU implementation of matrix transpose.
float *cpu_matrix_transpose(float *a_in, int m, int n){
    float *a_trans = (float*) malloc(sizeof(float) * n * m);
    for(int row = 0; row < n; row++){
        for(int col = 0; col < m; col++){
            a_trans[row * m + col] = a_in[col * n + row];
        }
    }
    return a_trans;
}



/**
    Kernel to transpose a Matrix.

    Parameters:
    - a_in: input vector holding the matrix to be transposed.
    - a_trans: output vector where to write the transposed matrix.
    - m, n: number of rows and columns respectively in matrix a_in.
*/
__global__ void matrix_transpose(float *a_in, float *a_trans, int m, int n){
    unsigned int idx_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_in < m * n){
        int row_in = ;
        int col_in = ;
        unsigned int idx_trans = ; // TODO: complete here
        a_trans[idx_trans] = a_in[idx_in];
    }
}



/**
 * Computes the transpose of the 2D matrix A of m rows and n columns.
 * Returns A_transposed, a matrix of n rows and m columns.
 */ 
 float* matrix_transpose_driver(float *A, int m, int n){
    float *dev_A, *dev_A_transposed, *A_transposed;
    HIP_CHECK_ERROR(hipMalloc(&dev_A, sizeof(float) * n * m));
    HIP_CHECK_ERROR(hipMalloc(&dev_A_transposed, sizeof(float) * n * m));
    HIP_CHECK_ERROR(hipMemcpy(dev_A, A, sizeof(float) * n * m, hipMemcpyHostToDevice));
    // allocate memory on CPU to store final result
    A_transposed = (float*) malloc(sizeof(float) * n * m);
    MALLOC_CHECK_ERROR(A_transposed);
    // setup kernel configuration
    unsigned int nBlocks = (n * m + NTHREADS - 1) / NTHREADS;
    ptimer_t kernel_timer;
    timer_start(&kernel_timer);
    matrix_transpose<<<nBlocks, NTHREADS>>>(dev_A, dev_A_transposed, m, n);
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    timer_stop(&kernel_timer);
    printf("'matrix_transpose' kernel execution time (ms): %.4f\n", timer_elapsed(kernel_timer));
    HIP_CHECK_ERROR(hipMemcpy(A_transposed, dev_A_transposed, sizeof(float) * n * m, hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    HIP_CHECK_ERROR(hipFree(dev_A));
    HIP_CHECK_ERROR(hipFree(dev_A_transposed));
    return A_transposed;
}



void test_performance(int n, int m){
    float *v = (float *) malloc(sizeof(float) * n * m);
    MALLOC_CHECK_ERROR(v);
    for(int i = 0; i < n * m; i++) v[i] = rand() % 3054;
    
    // compute transpose on CPU
    ptimer_t cpu_transpose_time;
    timer_start(&cpu_transpose_time);
    float *cpu_result = cpu_matrix_transpose(v, m, n);
    timer_stop(&cpu_transpose_time);
    printf("'cpu_matrix_transpose' execution time (ms): %.4f\n", timer_elapsed(cpu_transpose_time));
    // compute on gpu
    ptimer_t gpu_transpose_time;
    timer_start(&gpu_transpose_time);
    float *gpu_result = matrix_transpose_driver(v, m, n);
    timer_stop(&gpu_transpose_time);
    printf("'matrix_transpose_driver' execution time (ms): %.4f\n", timer_elapsed(gpu_transpose_time));
    
    for(int i = 0; i < n * m; i++)
        if(gpu_result[i] != cpu_result[i]){
            fprintf(stderr, "Performance: transpose is not correct.\n");
            exit(1);
        }
    printf("Performance test: all good.\n");
    free(gpu_result);
    free(cpu_result);
}



void test_correctness(void){
    int n = 3, m = 3;
    float A[] = {1, 2, 3,
                 4, 5, 6,
                 7, 8, 9};
    float A_check[] = 
                {1, 4, 7,
                 2, 5, 8,
                 3, 6, 9};
                    
    
    float *result = matrix_transpose_driver(A, m, n);
    // check the result is correct
    for(int i = 0; i < n*m; i++){
        if(A_check[i] != result[i]){
            fprintf(stderr, "Transpose is not correct.\n");
            free(result);
            exit(1);
        }
    }
    free(result);
    printf("Correctness test: all good.\n");
}



int main(int argc, char **argv){
    int n = 1e3;
    int m = 1e2;
    if (argc >= 3){
        n = atoi(argv[1]);
        m = atoi(argv[2]);
    }else{
        printf("Using defaults n = 1000, m = 100 for performance testing.\nTo change behaviour, %s [m n]\n\n", argv[0]);
    }
    // The following call is done to initialise HIP here and not to include HIP initialisation in subsequent calls, which we are timing.
    HIP_CHECK_ERROR(hipSetDevice(0));
    test_correctness();
    test_performance(n, m);
    return 0;
}
