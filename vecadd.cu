#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// timer function
double CPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

// CPU vector addition: z_h = x_h + y_h
void vecAdd(float* x_h, float* y_h, float* z_h, unsigned int n){
    double startTime = CPUTimer();

    for (unsigned int i = 0; i < n; i++){
        z_h[i] = x_h[i] + y_h[i];
    }

    double endTime = CPUTimer();
    printf("vecAdd on CPU: %.6f seconds\n", endTime - startTime);
}

// GPU vector addition

// main
int main(){
    cudaDeviceSynchronize(); 
    unsigned int n = 1024;

    // initialize arrays for x_h, y_h, and z_h
    float* x_h = (float*) malloc(sizeof(float)*n);
    for (unsigned int i=0; i < n; i++) x_h[i] =(float)rand()/(float)(RAND_MAX);
    float* y_h = (float*) malloc(sizeof(float)*n);
    for (unsigned int i=0; i < n; i++) y_h[i] =(float)rand()/(float)(RAND_MAX);
    float* z_h = (float*) calloc(n, sizeof(float));
    printf("Vector size %d\n", n * sizeof(z_h));

    // call the CPU vector addition
    vecAdd(x_h, y_h, z_h, n);

    // free host memory of arrays
    free(x_h);
    free(y_h);
    free(z_h);

    return 0;
}
