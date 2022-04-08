#include <stdlib.h>
#include <stdio.h>


void __cuda_check_error(cudaError_t err, const char *file, int line){
	if(err != cudaSuccess){
        fprintf(stderr, "CUDA error (%s:%d): %s\n", file, line, cudaGetErrorString(err));
        exit(1);
    }
}


#define CUDA_CHECK_ERROR(X)({\
	__cuda_check_error((X), __FILE__, __LINE__);\
})



int main(int argc, char** argv){
	int count;			// variable for number of devices
	int device;			// variable for active device id

	CUDA_CHECK_ERROR(cudaGetDeviceCount(&count));
	printf("\nFound %i devices\n\n", count);

	for (device = 0; device < count; device++){
		CUDA_CHECK_ERROR(cudaSetDevice(device));

		struct cudaDeviceProp p;
		CUDA_CHECK_ERROR(cudaGetDeviceProperties(&p, device));

		printf("Device %i : ", device);
		printf("%s ", p.name);
		printf("with %i SMs\n", p.multiProcessorCount);
	}
	
	printf("\n");
	return 0;
}
