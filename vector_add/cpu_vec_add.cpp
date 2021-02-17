#include <cstdlib>

void init_vec(float *v, int n){
    for(int i = 0; i < n; i++){
        v[i] = rand() % 100 * 0.3234f;
    }
}



void vector_add(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}



int main(void){
    int n = 100;
    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];
    init_vec(A, n);
    init_vec(B, n);
    vector_add(A, B, C, n);
    // do something with C ..
    delete[] A;
    delete[] B;
    delete[] C; 
    return 0;    
}
