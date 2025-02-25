#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h> 

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if (cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason:%s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}

double CPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

// CPU only matrix multiplication
void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    for (int row1 = 0; row1 < m; row1++){
        for (int col2 = 0; col2 < n; col2++){
            C_h[row1 * n + col2] = (float)0;
            for (int col1AndRow2 = 0; col1AndRow2 < k; col1AndRow2++){
                C_h[row1 * n + col2] += A_h[row1 * k + col1AndRow2] * B_h[col1AndRow2 * n + col2];
            }
        }
    }

    // print results
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", C_h[i * n + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    // ./sgemm <m> <k> <n>

    int m = atof(argv[1]);
    int k = atof(argv[2]);
    int n = atof(argv[3]);

    printf("Vector size of matrix A (m * k): %d\n", m * k);
    printf("Vector size of matrix B (k * n): %d\n", k * n);
    printf("Vector size of matrix C (m * n): %d\n", m * n);

    // initialize and populate arrays for A_h, B_h, and C_h with host memory
    float* A_h = (float*) malloc(sizeof(float)*(m * k));
    float* B_h = (float*) malloc(sizeof(float)*(k * n));
    float* C_h = (float*) calloc(m * n, sizeof(float));

    for (unsigned int i=0; i < n; i++){
        A_h[i] =(float)rand()/(float)(rand()%100/100.0);
        B_h[i] =(float)rand()/(float)(rand()%100/100.0);
    }

    // perform matrix multiplication
    basicSgemm_h(m, k, n, A_h, B_h, C_h);

    // free allocated memory
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}
