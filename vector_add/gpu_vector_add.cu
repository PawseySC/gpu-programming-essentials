#include <cstdlib>
#include <iostream>



#define CUDA_CHECK_ERROR(X)({\
    if((X) != cudaSuccess){\
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString((X)));\
        exit(1);\
    }\
})


// Initialise the vector v of n elements to random values
void init_vec(float *v, int n){
    for(int i = 0; i < n; i++){
        v[i] = rand() % 100 * 0.3234f;
    }
}


// kernel to perform vector addition
__global__ void vector_add(float *a, float *b, float *c, int n){
    unsigned int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int main(void){
    int n = 100;
    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];
    init_vec(A, n);
    init_vec(B, n);
    float *dev_A, *dev_B, *dev_C;
    CUDA_CHECK_ERROR(cudaMalloc(&dev_A, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMalloc(&dev_B, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMalloc(&dev_C, sizeof(float) * n));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_A, A, sizeof(float) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_B, B, sizeof(float) * n, cudaMemcpyHostToDevice));
    vector_add<<<1, n>>>(dev_A, dev_B, dev_C, n);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaMemcpy(C, dev_C, sizeof(float) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    
    // check the result is correct
    for(int i = 0; i < n; i++){
        if(C[i] != A[i] + B[i]){
            std::cerr << "Sum is not correct." << std::endl;
            cudaFree(dev_A);
            cudaFree(dev_B);
            cudaFree(dev_C);
            delete[] A;
            delete[] B;
            delete[] C;
            return 1;
        }
    }
    CUDA_CHECK_ERROR(cudaFree(dev_A));
    CUDA_CHECK_ERROR(cudaFree(dev_B));
    CUDA_CHECK_ERROR(cudaFree(dev_C));
    delete[] A;
    delete[] B;
    delete[] C;
    std::cout << "All good." << std::endl;
    return 0;
}
