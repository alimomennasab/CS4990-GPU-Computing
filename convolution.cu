#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h> 
#include <opencv2/opencv.hpp>

# define FILTER_RADIUS 2
#define TILE_DIM 32 
# define FILTER_DIM (2 * FILTER_RADIUS + 1)

// constant average filter
const float F_h[FILTER_DIM][FILTER_DIM] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};
__constant__ float F_d[FILTER_DIM][FILTER_DIM];

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

// verification between two cv::Mat images
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols){
    const float relativeTolerance = 1e-2;
    for (int i = 0; i < nRows; i++){
        for (int j = 0; j < nCols; j++){
            float relativeError = ((float)answer1.at<unsigned char>(i, j) - (float)answer2.at<unsigned char>(i, j)) / 255;
            if (fabs(relativeError) > relativeTolerance){
                printf("Failed at (%d, %d) with relative error %f\n", i, j, relativeError);
                printf("    answer1.at<unsigned char>(%d, %d) = %u\n", i, j, answer1.at<unsigned char>(i, j));
                printf("    answer2.at<unsigned char>(%d, %d) = %u\n\n", i, j, answer2.at<unsigned char>(i, j));
                return false;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return true;
}

// CPU implementation of image blur with average box filter F_h
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){
    for (int rowIdx = 0; rowIdx < nRows; rowIdx++){
        for (int colIdx = 0; colIdx < nCols; colIdx++){
            float sumPixVal = 0.0f;

            for (int blurRowOffset = -1 * FILTER_RADIUS; blurRowOffset < (FILTER_RADIUS + 1); blurRowOffset++){
                for (int blurColOffset = -1 * FILTER_RADIUS; blurColOffset < (FILTER_RADIUS + 1); blurColOffset++){
                    int curRowIdx = rowIdx + blurRowOffset;
                    int curColIdx = colIdx + blurColOffset;

                    if ((curRowIdx >= 0) && (curRowIdx < nRows) && (curColIdx >= 0) && (curColIdx < nCols)){
                        sumPixVal += (Pin_Mat_h.at<unsigned char>(curRowIdx, curColIdx) / 255.0) * F_h[blurRowOffset + FILTER_RADIUS][blurColOffset + FILTER_RADIUS];
                    }
                }
            }
            // compute the average
            Pout_Mat_h.at<unsigned char>(rowIdx, colIdx) = (unsigned char)(sumPixVal * 255.0f);
        }
    }
}

// CUDA kernel of image blur with average box filter F_d
__global__ void blurImage_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height){
    int colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (colIdx < width && rowIdx < height){
        float sumPixVal = 0.0f;

        for (int blurRowOffset = -1 * FILTER_RADIUS; blurRowOffset < (FILTER_RADIUS + 1); blurRowOffset++){
            for (int blurColOffset = -1 * FILTER_RADIUS; blurColOffset < (FILTER_RADIUS + 1); blurColOffset++){
                int curRowIdx = rowIdx + blurRowOffset;
                int curColIdx = colIdx + blurColOffset;

                if ( (curRowIdx >= 0) && (curRowIdx < height) && (curColIdx >= 0) && (curColIdx < width)){
                    sumPixVal += (Pin[curRowIdx * width + curColIdx] / 255.0) * F_d[blurRowOffset + FILTER_RADIUS][blurColOffset + FILTER_RADIUS];
                }
            }

        }
        // compute the average
        Pout[rowIdx * width + colIdx] = (unsigned char)(sumPixVal * 255.0f);
    }
}

// GPU implementation of image blur with average box filter
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){
    // allocate device memory for the input and output image
    unsigned char * Pin_d, * Pout_d;
    CHECK(cudaMalloc((void**)&Pin_d, nRows * nCols * sizeof(unsigned char)));
    CHECK(cudaMalloc((void**)&Pout_d, nRows * nCols * sizeof(unsigned char)));

    // copy the input image from host to device, and the filter
    CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(F_d, F_h, sizeof(F_h)));

    // set the kernel launch parameters
    dim3 blockDim(32, 32);
    unsigned int gridDimX = (nCols + blockDim.x - 1) / blockDim.x;  // number of blocks in x-direction (cols)
    unsigned int gridDimY = (nRows + blockDim.y - 1) / blockDim.y;  // number of blocks in y-direction (rows)
    dim3 gridDim = {gridDimX, gridDimY};
    blurImage_Kernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, nCols, nRows);

    // copy the output image from device to host
    CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows * nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // free device memory
    CHECK(cudaFree(Pin_d));
    CHECK(cudaFree(Pout_d));

}

// optimized CUDA kernel of tiled image blur using the average box filter from constant memory
__global__ void blurImage_tiled_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height) {
    __shared__ float tile[TILE_DIM + 2 * FILTER_RADIUS][TILE_DIM + 2 * FILTER_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_DIM + ty;  // output pixel row
    int col_o = blockIdx.x * TILE_DIM + tx;  // output pixel col

    // Loop over shared memory tile with coarsening to load the tile and the padding 
    for (int i = ty; i < TILE_DIM + 2 * FILTER_RADIUS; i += blockDim.y) {
        for (int j = tx; j < TILE_DIM + 2 * FILTER_RADIUS; j += blockDim.x) {
            int row_i = blockIdx.y * TILE_DIM + i - FILTER_RADIUS;
            int col_i = blockIdx.x * TILE_DIM + j - FILTER_RADIUS;

            if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
                tile[i][j] = Pin[row_i * width + col_i] / 255.0f;
            } else {
                tile[i][j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // compute blurred pixel
    if (row_o < height && col_o < width) {
        float sumPixVal = 0.0f;

        for (int fy = 0; fy < FILTER_DIM; fy++) {
            for (int fx = 0; fx < FILTER_DIM; fx++) {
                sumPixVal += tile[ty + fy][tx + fx] * F_d[fy][fx];
            }
        }

        // Convert back to byte and store
        Pout[row_o * width + col_o] = (unsigned char)(sumPixVal * 255.0f);

    }
}

// GPU implementation of image blur with shared memory tiled convolutions with average box filter from constant memory
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){
    // allocate device memory for the input and output image
    unsigned char * Pin_d, * Pout_d;
    CHECK(cudaMalloc((void**)&Pin_d, nRows * nCols * sizeof(unsigned char)));
    CHECK(cudaMalloc((void**)&Pout_d, nRows * nCols * sizeof(unsigned char)));

    // copy the input image from host to device, and the filter
    CHECK(cudaMemcpy(Pin_d, Pin_Mat_h.data, nRows * nCols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(F_d, F_h, sizeof(F_h)));

    // set the kernel launch parameters
    dim3 blockDim(32, 32);
    unsigned int gridDimX = (nCols + blockDim.x - 1) / blockDim.x;  // number of blocks in x-direction (cols)
    unsigned int gridDimY = (nRows + blockDim.y - 1) / blockDim.y;  // number of blocks in y-direction (rows)
    dim3 gridDim = {gridDimX, gridDimY};
    blurImage_tiled_Kernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, nCols, nRows);

    // copy the output image from device to host
    CHECK(cudaMemcpy(Pout_Mat_h.data, Pout_d, nRows * nCols * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // free device memory
    CHECK(cudaFree(Pin_d));
    CHECK(cudaFree(Pout_d));
}

// main
int main(int argc, char** argv){
    // ./convolution <inputImg.jpg>
    CHECK(cudaDeviceSynchronize());
    double startTime, endTime;

    // load filename from args
    char *fileName = argv[1];
    if (fileName == NULL){
        printf("Error: No input image file name provided.\n");
        return -1;
    }

    // use OpenCV to load a grayscale image
    cv::Mat grayImg = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
    if (grayImg.empty()) return -1;

    // obtain image's height, width, and number of channels
    unsigned int nRows = grayImg.rows, nCols = grayImg.cols, nChannels = grayImg.channels();

    // use OpenCV's blur
    cv::Mat blurredImg_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = CPUTimer();
    cv::blur(grayImg, blurredImg_opencv, cv::Size(2 * FILTER_RADIUS + 1, 2 * FILTER_RADIUS + 1), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    endTime = CPUTimer();
    printf("openCV's blur (CPU): %f s\n\n", endTime-startTime);

    // CPU blurring
    cv::Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = CPUTimer();
    blurImage_h(blurredImg_cpu, grayImg, nRows, nCols);
    endTime = CPUTimer();
    printf("blurImage on CPU: %f s\n\n", endTime-startTime);

    // GPU blurring that calls the kernel
    cv::Mat blurredImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = CPUTimer();
    blurImage_d(blurredImg_gpu, grayImg, nRows, nCols);
    endTime = CPUTimer();
    printf("blurImage on GPU: %f s\n\n", endTime-startTime);
    
    // GPU blurring that calls the shared-memory tiled convolution kernel
    cv::Mat blurredImg_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = CPUTimer();
    blurImage_tiled_d(blurredImg_tiled_gpu, grayImg, nRows, nCols);
    endTime = CPUTimer();
    printf("(tiled)blurImage on GPU: %f s\n\n", endTime-startTime);

    // saved the blurred image to disk
    bool check = cv::imwrite("./blurredImg_opencv.jpg", blurredImg_opencv);
    if(check == false) {printf("Error!\n"); return -1; }

    check = cv::imwrite("./blurredImg_cpu.jpg", blurredImg_cpu);
    if(check == false) {printf("Error!\n"); return -1; }

    check = cv::imwrite("./blurredImg_gpu.jpg", blurredImg_gpu);
    if(check == false) {printf("Error!\n"); return -1; }

    check = cv::imwrite("./blurredImg_tiled_gpu.jpg", blurredImg_tiled_gpu);
    if(check == false) {printf("Error!\n"); return -1; }

    // verify the results
    printf("Verifying CPU blurring results: " );
    verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);

    printf("Verifying GPU blurring results: ");
    verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);

    printf("Verifying tiled GPU blurring results: ");
    verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);

    return 0;

}
