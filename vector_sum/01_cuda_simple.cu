#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK_ERROR(X)({\
    if((X) != cudaSuccess){\
        fprintf(stderr, "ERROR %d (%s:%d): %s\n", (X), __FILE__, __LINE__, cudaGetErrorString((X)));\
        exit(1);\
    }\
})

#define NTHREADS 1024 



__global__ void vector_sum(unsigned char *values, unsigned int nitems, unsigned long long* result){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < nitems) atomicAdd(result, values[idx]);  
}



int main(int argc, char **argv){
    
    unsigned int nitems = 1e9; 
    unsigned char *values = (unsigned char*) malloc(sizeof(unsigned int) * nitems);
    if(!values){
        fprintf(stderr, "Error while allocating memory\n");
        return -1;
    }
    // Initialise the vector of n elements to random values
    unsigned long long correct_result = 0;
    for(int i = 0; i < nitems; i++){
        values[i] = rand() % 256;
        correct_result += values[i];
    }

    unsigned long long sum = 0ull;
    unsigned long long *dev_sum;
    unsigned char *dev_values;
    CUDA_CHECK_ERROR(cudaMalloc(&dev_values, sizeof(unsigned char) * nitems));
    CUDA_CHECK_ERROR(cudaMalloc(&dev_sum, sizeof(unsigned long long)));
    CUDA_CHECK_ERROR(cudaMemset(dev_sum, 0, sizeof(unsigned long long)));
    CUDA_CHECK_ERROR(cudaMemcpy(dev_values, values, sizeof(unsigned char) * nitems, cudaMemcpyHostToDevice));
    unsigned int nblocks = (nitems + NTHREADS - 1) / NTHREADS;
    cudaEvent_t start, stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));
    CUDA_CHECK_ERROR(cudaEventRecord(start)); 
    vector_sum<<<nblocks, NTHREADS>>>(dev_values, nitems, dev_sum);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaEventRecord(stop)); 
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaMemcpy(&sum, dev_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    float time_spent;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&time_spent, start, stop));
    printf("Result: %llu - Time elapsed: %f\n", sum, time_spent/1000.0f);
    if(correct_result != sum) {
        fprintf(stderr, "Error: sum is not correct, should be %llu\n", correct_result);
        return 1;
    }
    return 0;

}
