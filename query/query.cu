#include <stdlib.h>
#include <stdio.h>


#define CUDA_CHECK_ERROR(X)({\
    if((X) != cudaSuccess){\
        fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString((X)));\
        exit(1);\
    }\
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
