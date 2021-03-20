#ifndef __ARRAY_H__
#define __ARRAY_H__



// prints start and end of float array
void print_array_terse(float* array, int length, int num){
	if (length < 2 * num)
        num = length/2;
	for (int i=0; i<num; i++){
		printf("%5.2f ",array[i]);
	}
	printf("... ");
    for (int i=length-num; i<length; i++){
        printf("%5.2f ",array[i]);
    }
    printf("\n");
}


#endif