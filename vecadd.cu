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
    printf("%-50s %10.6f s\n\n", "vecAdd on CPU:", endTime - startTime); 
}

// GPU vector addition
__global__ void vecAddKernel(float* x_d, float* y_d, float* z_d, unsigned int n){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // unique global thread index
    if (i < n){
        z_d[i] = x_d[i] + y_d[i];
    }
}

// main
int main(){
    CHECK(cudaDeviceSynchronize()); 
    unsigned int n = 16777216; 

    // initialize arrays for x_h, y_h, and z_h with host memory
    float* x_h = (float*) malloc(sizeof(float)*n);
    for (unsigned int i=0; i < n; i++) x_h[i] =(float)rand()/(float)(RAND_MAX);
    float* y_h = (float*) malloc(sizeof(float)*n);
    for (unsigned int i=0; i < n; i++) y_h[i] =(float)rand()/(float)(RAND_MAX);
    float* z_h = (float*) calloc(n, sizeof(float));
    printf("Vector size %d\n", n);

    // call CPU vector addition
    vecAdd(x_h, y_h, z_h, n);

    // allocate device memory on GPU for arrays x_d, y_d, z_d
    double startTimeGPUAddition = CPUTimer();

    float *x_d,  *y_d, *z_d;
    double startTimeCudaMalloc = CPUTimer();
    CHECK(cudaMalloc((void**)&x_d, sizeof(float)*n));
    CHECK(cudaMalloc((void**)&y_d, sizeof(float)*n));
    CHECK(cudaMalloc((void**)&z_d, sizeof(float)*n));
    double endTimeCudaMalloc = CPUTimer();
    printf("%-50s %10.6f s\n", "     cudaMalloc:", endTimeCudaMalloc - startTimeCudaMalloc); 

    // copy x_h and y_h to x_d and y_d
    double startTimeCopyHostArrs = CPUTimer();
    CHECK(cudaMemcpy(x_d, x_h, sizeof(float)*n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(y_d, y_h, sizeof(float)*n, cudaMemcpyHostToDevice));
    double endTimeCopyHostArrs = CPUTimer();
    printf("%-50s %10.6f s\n", "     cudaMemcpy:", endTimeCopyHostArrs - startTimeCopyHostArrs); 

    // call GPU vector addition
    dim3 blockSize(512, 1, 1); 
    dim3 gridSize(ceil(n / 512.0), 1, 1);

    double startTimeKernelCall = CPUTimer();
    vecAddKernel<<<gridSize, blockSize>>>(x_d, y_d, z_d, n);
    double endTimeKernelCall = CPUTimer();
    printf("%-50s %10.6f s\n", "     vecAddKernel<<<(32768,1,1),(512,1,1)>>>:", endTimeKernelCall - startTimeKernelCall); 

    // copy GPU addition results to host memory
    double startTimeCudaMemcpy = CPUTimer();
    cudaMemcpy(z_h, z_d, sizeof(float)*n, cudaMemcpyDeviceToHost);
    double endTimeCudaMemcpy = CPUTimer();
    printf("%-50s %10.6f s\n", "     cudaMemcpy:", endTimeCudaMemcpy - startTimeCudaMemcpy); 

    double endTimeGPUAddition = CPUTimer();
    printf("%-50s %10.6f s\n", "vecAdd on GPU:", endTimeGPUAddition - startTimeGPUAddition);


    // free device memory of device arrays 
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

    // free host memory of host arrays
    free(x_h);
    free(y_h);
    free(z_h);

    return 0;
}
