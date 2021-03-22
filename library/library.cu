#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "../common/array.h"


// prints error if detected and exits 
#define CUDA_CHECK_ERROR(X)({\
    if((X) != cudaSuccess){\
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString((X)));\
        exit(1);\
    }\
})

// detects cublas non-sucess status and exits
#define CUBLAS_CHECK_ERROR(X)({\
    if ((X) != CUBLAS_STATUS_SUCCESS){\
        fprintf(stderr, "CUBLAS error (%s:%d): %i\n", __FILE__, __LINE__, (X));\
        exit(1);\
    }\
})

#define MALLOC_CHECK_ERROR(X)({\
    if ((X) == 0){\
        fprintf(stderr, "Malloc error (%s:%d): %i\n", __FILE__, __LINE__, (X));\
        exit(1);\
    }\
})



// add two vectors
int main(int argc, char** argv){
	// variable declarations
    cublasHandle_t handle;           // variable for cublas handle
    int device;                      // current device id
    struct cudaDeviceProp prop;      // current device properties
	float* hostArrayA;                 // pointer for array A in host memory
    float* hostArrayB;                 // pointer for array B in host memory
	float* deviceArrayA;               // pointer for array A in device memory
    float* deviceArrayB;               // pointer for array B in device memory
	int length = 262144;             // length of array
    int size = length*sizeof(float);   // size of array in bytes

    // get device properties
    CUDA_CHECK_ERROR(cudaGetDevice(&device));
    CUDA_CHECK_ERROR(cudaGetDeviceProperties(&prop, device));
    printf("\nDevice properties: using %s\n\n",prop.name);

	// allocate host memory
	hostArrayA = (float*) malloc(sizeof(float) * length);
    hostArrayB = (float*) malloc(sizeof(float) * length);
    MALLOC_CHECK_ERROR(hostArrayA && hostArrayB);

	// allocate device memory
	CUDA_CHECK_ERROR(cudaMalloc(&deviceArrayA, size));
    CUDA_CHECK_ERROR(cudaMalloc(&deviceArrayB, size));
    
	// initialise host memory
	for(int i=0; i<length; i++){
		hostArrayA[i] = i;
        hostArrayB[i] = 1;
	}

    // print host memory values for all arrays
	printf("Array A: ");
	print_array_terse(hostArrayA, length, 8);
    printf("Array B: ");
    print_array_terse(hostArrayB, length, 8);

	// prepare cuBLAS context
    CUBLAS_CHECK_ERROR(cublasCreate(&handle));

	// copy host to device for arrays A and B
	CUBLAS_CHECK_ERROR(cublasSetVector(length, sizeof(float), hostArrayA, 1, deviceArrayA, 1));
   
    // TODO: use cublasSetVector to copy array B to the device
    printf("\nCopied array A and B to device\n\n");

    // perform B = 1*A + B using cublas
	const float c = 1.0f;
    // TODO: use cublasSaxpy to add arrays A and B together
    
    printf("Performed B = A + B using cublas\n\n");

	// copy device to host for array B
	// TODO: use cublasGetVector copy array B back to the host
    printf("Copied array B from device\n\n");

    // destroy cuBLAS context
    CUBLAS_CHECK_ERROR(cublasDestroy(handle));

	// print host memory values for array C
    printf("Array B: ");
    print_array_terse(hostArrayB, length, 8);

	// free device memory
    CUDA_CHECK_ERROR(cudaFree(deviceArrayA));
    CUDA_CHECK_ERROR(cudaFree(deviceArrayB));
    
	// free host memory
	free(hostArrayA);
    free(hostArrayB);
    printf("\nFreed device and host memory\n\n");

	return 0;
}