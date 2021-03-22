#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MALLOC_CHECK_ERROR(X)({\
    if ((X) == 0){\
        fprintf(stderr, "Malloc error (%s:%d): %i\n", __FILE__, __LINE__, (X));\
        exit(1);\
    }\
})



/*
 * Sums all the elements of a vector v of n elements, where n is a power of 2,
 * using the parallel reduction algorithm.
 */
int reduction(const int *v, int n){
    // This is an inplace algorithm. You don't want to modify the input data,
    // so create a work copy first.
    int *work_copy = (int *) malloc(sizeof(int) * n);
    MALLOC_CHECK_ERROR(work_copy);
    memcpy(work_copy, v, sizeof(int) * n);
    // TODO: complete the following code
    for(int stride = ; stride >= ; stride = ){
        for (int thread = 0; thread < ; thread = thread + 1){
            work_copy[thread] = work_copy[thread] + ;
        }
    }
    int result = work_copy[0];
    free(work_copy);
    return result;
}



void test_reduction(){
    const int v1[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    if(reduction(v1, 8) != 8){
        fprintf(stderr, "Error, reduction on v1 is not correct!\n");
        exit(1);
    } 


    const int v2[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    if(reduction(v2, 16) != 136){
        fprintf(stderr, "Error, reduction on v2 is not correct!\n");
        exit(1);
    }
}



int main(void){
    test_reduction();
    printf("Reduction: it works!\n");
    return 0;
}