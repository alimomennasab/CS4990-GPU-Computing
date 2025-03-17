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
    printf("basicSgemm_h on CPU: %.6f s\n\n", endTime - startTime);
}

// CUDA kernel where each thread computes one output matrix element
__global__ void matrixMulKernel_1thread1element (int m, int k, int n, const float *A_d, const float *B_d, float* C_d){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n){
        float sum = 0.0f;
        for (unsigned int i = 0; i < k; ++i) {
            sum += A_d[row * k + i] * B_d[i * n + col];
        }
        C_d[row * n + col] = sum;
    }
}

// A CUDA kernel for a “tiled” version of matrix multiplication, which uses dynamically allocated space in shared memory. 
// Here we assume each thread calculates one element of the output matrix
__global__ void matrixMulKernel_tiled(int m, int k, int n, const float *A_d, const float 
    *B_d, float* C_d, unsigned Adz_sz, unsigned Bdz_sz){

}

// host function allocating & freeing & copying memory/calling/timing for matrixMulKernel_1thread1element
void basicSgemm_d_1thread1element (int m, int k, int n, const float *A_h, const float *B_h, float* C_h){
    printf("1thread1element on GPU: \n");
    double startTotalTime = CPUTimer();

    // allocate device memory on GPU for arrays A_d, B_d, C_d
    float *A_d,  *B_d, *C_d;
    double startTimeCudaMalloc = CPUTimer();
    CHECK(cudaMalloc((void**)&A_d, sizeof(float)*(m * k)));
    CHECK(cudaMalloc((void**)&B_d, sizeof(float)*(k * n)));
    CHECK(cudaMalloc((void**)&C_d, sizeof(float)*(m * n)));
    double endTimeCudaMalloc = CPUTimer();
    printf("    1thread1element cudaMalloc: %.6f s\n", endTimeCudaMalloc - startTimeCudaMalloc);

    // copy A_h and B_h to A_d and B_d
    double startTimeCudaMemcpy = CPUTimer();
    CHECK(cudaMemcpy(A_d, A_h, sizeof(float)*(m*k), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeof(float)*(k*n), cudaMemcpyHostToDevice));
    double endTimeCudaMemcpy = CPUTimer();
    printf("    1thread1element cudaMemcpy: %.6f s\n", endTimeCudaMemcpy - startTimeCudaMemcpy);

    // calling matrixMulKernel_1thread1element kernel
    dim3 blockDim(32, 32);
    unsigned int gridDimX = (n + blockDim.x - 1) / blockDim.x;  // number of blocks in x-direction (cols)
    unsigned int gridDimY = (m + blockDim.y - 1) / blockDim.y;  // number of blocks in y-direction (rows)
    dim3 gridDim = {gridDimX, gridDimY};

    double startTimeKernelCall = CPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(m, k, n, A_d, B_d, C_d);
    double endTimeKernelCall = CPUTimer();
    printf("    matrixMulKernel_1thread1element<<<(%d,%d,1),(%d,%d,1)>>> call time: %.6f s\n", 
        gridDim.x, gridDim.y, blockDim.x, blockDim.y, endTimeKernelCall - startTimeKernelCall);

    // copy GPU matmul results to host memory
    double startTimeCudaMemcpyResults = CPUTimer();
    cudaMemcpy(C_h, C_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    double endTimeCudaMemcpyResults = CPUTimer();
    printf("    1thread1element results cudaMemcpy: %.6f s\n", endTimeCudaMemcpyResults - startTimeCudaMemcpyResults);

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // timing results
    double endTotalTime = CPUTimer();
    printf("    1thread1element total time: %.6f s\n", endTotalTime - startTotalTime);
}

// host function allocating & freeing & copying memory/calling/timing formatrixMulKernel_tiled()
void basicSgemm_d_tiled (int m, int k, int n, const float *A_h, const float *B_h, float*C_h){

}

// verification between GPU and CPU result matrices
bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols){
    float epsilon = 0.01f;
    int wrongCount = 0;

    for (unsigned int i = 0; i < nRows; i++) {
        for (unsigned int j = 0; j < nCols; j++) {
            float cpuValue = CPU_Answer[i * nCols + j];
            float gpuValue = GPU_Answer[i * nCols + j];
            float difference = fabs(cpuValue - gpuValue);
            if (difference > epsilon) {
                printf("cpuValue[%d][%d] = %f doesn't match gpuValue[%d][%d] = %f \n", i, j, cpuValue, i, j, gpuValue);
                printf("difference: %f \n", difference);
                wrongCount++;
            }
        }
    }
    if (wrongCount != 0){
        printf("Mismatching CPU/GPU result matrix elements count: %d \n\n", wrongCount);
        return false;
    }
    return true;
}

// main
int main(int argc, char *argv[]) {
    // ./sgemm <m> <k> <n>
    CHECK(cudaDeviceSynchronize()); 

    int m = atof(argv[1]);
    int k = atof(argv[2]);
    int n = atof(argv[3]);

    printf("Vector size of matrix A (m * k): %d * %d = %d\n", m, k, m * k);
    printf("Vector size of matrix B (k * n): %d * %d = %d\n", k, n, k * n);
    printf("Vector size of matrix C (m * n): %d * %d = %d\n\n", m, n, m * n);

    // initialize and populate arrays for A_h, B_h, and C_h with host memory
    float* A_h = (float*) malloc(sizeof(float)*(m * k));
    float* B_h = (float*) malloc(sizeof(float)*(k * n));
    float* C_h = (float*) calloc(m * n, sizeof(float));
    float* C_h_gpu_answer = (float*) calloc(m * n, sizeof(float));

    for (unsigned int i = 0; i < m * k; i++) {
        A_h[i] = rand()%100/100.0f;
    }
    for (unsigned int i = 0; i < k * n; i++) {
        B_h[i] = rand()%100/100.0f;
    }

    // perform CPU matrix multiplication
    basicSgemm_h(m, k, n, A_h, B_h, C_h);

    // perform GPU matrix multiplication
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_h_gpu_answer);
    if (verify(C_h, C_h_gpu_answer, m, n)){
        printf("Verification successful: GPU and CPU result matrices match\n\n");
    } else {
        printf("Verification failed: GPU and CPU result matrices don't match\n\n");
    }

    // perform GPU tiled matrix multiplication

    // free allocated host memory
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_h_gpu_answer);

    return 0;
}
