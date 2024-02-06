#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>

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
inline bool compare_float(float a, float b){
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


// kernel to perform vector addition
__global__ void vector_add(float *a, float *b, float *c){
    unsigned int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int main(void){
    const unsigned int n = 100;
    float *A = (float*) malloc(n * sizeof(float));
    float *B = (float*) malloc(n * sizeof(float));
    float *C = (float*) malloc(n * sizeof(float));
    MALLOC_CHECK_ERROR(A && B && C);
    init_vec(A, n);
    init_vec(B, n);
    float *dev_A, *dev_B, *dev_C;
    CUDA_CHECK_ERROR(hipMalloc(&dev_A, sizeof(float) * n));
    CUDA_CHECK_ERROR(hipMalloc(&dev_B, sizeof(float) * n));
    CUDA_CHECK_ERROR(hipMalloc(&dev_C, sizeof(float) * n));
    CUDA_CHECK_ERROR(hipMemcpy(dev_A, A, sizeof(float) * n, hipMemcpyHostToDevice));
    CUDA_CHECK_ERROR(hipMemcpy(dev_B, B, sizeof(float) * n, hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(vector_add, 1, n, 0, 0, dev_A, dev_B, dev_C);

    CUDA_CHECK_ERROR(hipGetLastError());
    // some errors happened during kernel execution may show up only after device synchronization
    CUDA_CHECK_ERROR(hipDeviceSynchronize()); 
    CUDA_CHECK_ERROR(hipMemcpy(C, dev_C, sizeof(float) * n, hipMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(hipDeviceSynchronize());
    
    // check the result is correct
    bool sums_equal = true;
    for(int i = 0; i < n; i++){
        sums_equal = compare_float(C[i], A[i] + B[i]);
        if(!sums_equal) break;
    }
    CUDA_CHECK_ERROR(hipFree(dev_A));
    CUDA_CHECK_ERROR(hipFree(dev_B));
    CUDA_CHECK_ERROR(hipFree(dev_C));
    free(A);
    free(B);
    free(C);
    if(sums_equal){
        printf("All good.\n");
        return EXIT_SUCCESS;
    }else{
        fprintf(stderr, "Sum is not correct.\n");
        return EXIT_FAILURE;
    }
    
}
