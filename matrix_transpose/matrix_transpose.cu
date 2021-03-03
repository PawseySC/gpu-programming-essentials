#include <cstdlib>
#include <iostream>


#define NTHREADS 1024
#define CUDA_CHECK_ERROR(X)({\
    if((X) != cudaSuccess){\
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString((X)));\
        exit(1);\
    }\
})


/**
    Kernel to transpose a Matrix.

    Parameters:
    - a_in: input vector holding the matrix to be transposed.
    - a_trans: output vector where to write the transposed matrix.
    - m, n: number of rows and columns respectively in matrix a_in.
*/
__global__ void matrix_transpose(float *a_in, float *a_trans, int m, int n){
    unsigned int idx_in = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx_in <  /* TODO: complete here */){
    
        unsigned int idx_trans =  // TODO: complete here
        a_trans[idx_trans] = a_in[idx_in];
    }
}


/**
 * Computes the transpose of the 2D matrix A of m rows and n columns.
 * Returns A_transposed, a matrix of n rows and m columns.
 */ 
float* matrix_transpose_driver(float *A, int m, int n){
    float *dev_A, *dev_A_transposed, *A_transposed;
    CUDA_CHECK_ERROR(cudaMalloc(&dev_A, sizeof(float) * n * m));
    CUDA_CHECK_ERROR(cudaMalloc(&dev_A_transposed, sizeof(float) * n * m));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_A, A, sizeof(float) * n * m, cudaMemcpyHostToDevice));
    // allocate memory on CPU to store final result
    A_transposed = new float[n * m];
    // setup kernel configuration
    unsigned int nBlocks = (n * m + NTHREADS - 1) / NTHREADS;
    matrix_transpose<<<nBlocks, NTHREADS>>>(dev_A, dev_A_transposed, m, n);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaMemcpy(A_transposed, dev_A_transposed, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaFree(dev_A));
    CUDA_CHECK_ERROR(cudaFree(dev_A_transposed));
    return A_transposed;
}



int main(void){
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
            std::cerr << "Transpose is not correct." << std::endl;
            delete[] result;
            return 1;
        }
    }
    delete[] result;
    std::cout << "All good." << std::endl;
    return 0;
}
