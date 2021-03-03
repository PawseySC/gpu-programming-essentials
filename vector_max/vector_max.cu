#include <cstdlib>
#include <iostream>
#include <chrono>



#define NTHREADS 1024
#define CUDA_CHECK_ERROR(X)({\
    if((X) != cudaSuccess){\
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString((X)));\
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
    // TODO: set the max as the first element of the array
    
    // setup kernel configuration
    unsigned int nBlocks = (n + NTHREADS - 1) / NTHREADS;
    vector_max<<<nBlocks, NTHREADS>>>(dev_v, dev_max, n);
    CUDA_CHECK_ERROR(cudaGetLastError());
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
        std::cerr << "Max is not correct." << std::endl;
        exit(1);
    }
    std::cout << "Correctness: all good." << std::endl;
}



void test_performance(void){
    int n = 1e9;
    int *v = new int[n];
    int max_check = -1;
    for(int i = 0; i < n; i++){
        v[i] = rand() % 3054;
        if(v[i] > max_check) max_check = v[i];
    }
    auto start = std::chrono::system_clock::now();
    int result = vector_max_driver(v, n);
    auto end = std::chrono::system_clock::now();
    if(max_check != result){
        std::cerr << "Performance: max is not correct." << std::endl;
        exit(1);
    }
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Performance: all good. Time: " << elapsed << " ms." << std::endl;
}



int main(void){
    test_correctness();
    test_performance();
    return 0;
}
