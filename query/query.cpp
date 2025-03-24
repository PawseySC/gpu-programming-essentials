#include <stdlib.h>
#include <stdio.h>
#include "hip/hip_runtime.h"

void __hip_check_error(hipError_t err, const char *file, int line){
	if(err != hipSuccess){
        fprintf(stderr, "HIP error (%s:%d): %s\n", file, line, hipGetErrorString(err));
        exit(1);
    }
}


#define HIP_CHECK_ERROR(X)({\
	__hip_check_error((X), __FILE__, __LINE__);\
})



int main(int argc, char** argv){
	int count;			// variable for number of devices
	int device;			// variable for active device id

	HIP_CHECK_ERROR(hipGetDeviceCount(&count));
	printf("\nFound %i devices\n\n", count);

	for (device = 0; device < count; device++){
		HIP_CHECK_ERROR(hipSetDevice(device));

		struct hipDeviceProp_t p;
		HIP_CHECK_ERROR(hipGetDeviceProperties(&p, device));

		printf("Device %i : ", device);
		printf("%s ", p.name);
		printf("with %i compute units.\n", p.multiProcessorCount);
	}
	
	printf("\n");
	return 0;
}
