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
    double startTime = CPUTimer();
    for (int row1 = 0; row1 < m; row1++){
        for (int col2 = 0; col2 < n; col2++){
            C_h[row1 * n + col2] = (float)0;
            for (int col1AndRow2 = 0; col1AndRow2 < k; col1AndRow2++){
                C_h[row1 * n + col2] += A_h[row1 * k + col1AndRow2] * B_h[col1AndRow2 * n + col2];
            }
        }
    }

    double endTime = CPUTimer();

    printf("basicSgemm_h on CPU: %.6f s\n", endTime - startTime);

    // print results
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", C_h[i * n + j]);
    //     }
    //     printf("\n");
    // }
}

// CUDA kernel where each thread computes one output matrix element, 
// and function for device allocation and free, memcpy, and calling the kernel
__global__ void matrixMulKernel_1thread1element (int m, int k, int n, const float *A_d, const float *B_d, float* C_d){
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < m && col < n){
        float sum = 0.0f;
        for (unsigned int i = 0; i < k; ++i) {
            sum += A_d[row * k + i] * B_d[i * n + col];
        }
        C_d[row * n + col] = sum;
    }

}

void basicSgemm_d_1thread1element (int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    // allocate device memory on GPU for arrays A_d, B_d, C_d
    float *A_d,  *B_d, *C_d;
    CHECK(cudaMalloc((void**)&A_d, sizeof(float)*(m * k)));
    CHECK(cudaMalloc((void**)&B_d, sizeof(float)*(k * n)));
    CHECK(cudaMalloc((void**)&C_d, sizeof(float)*(m * n)));

    // copy A_h and B_h to A_d and B_d
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float)*(m*k), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float)*(k*n), cudaMemcpyHostToDevice));

    // calling matrixMulKernel_1thread1element kernel
    matrixMulKernel_1thread1element<<<(ceil(n / 512.0), 1, 1), (512, 1, 1)>>>(m, k, n, A_d, B_d, C_d);

    // copy GPU matmul results to host memory
    cudaMemcpy(C_h, C_d, sizeof(float)*n, cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

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
