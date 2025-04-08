#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h> 
#include <opencv2/opencv.hpp>

# define FILTER_RADIUS 2

// constant average filter
const float F_h[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};

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
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsighed int nCols){
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

// CPU implementation of image blur with average box filter
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows unsigned int nCols){

}

// CUDA kernel of image blur with average box filter
__global__void blurImage_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height){

}

// GPU implementation of image blur with average box filter
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){

}

// optimized CUDA kernel of image blur using the average box filter from constant memory
__global__ void blurImage_tiled_Kernel(unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height){

}

// GPU implementation of image blur with shared memory tiled convolutions with average box filter from constant memory
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols){

}

// main
int main(int argc, char** argv){
    // parse the command line args to extrac the filename of the input iamge specified by the user

    // error handling when file can't be found/loaded

    // load the input image dynamically at runtime instead of using a hardcoded filename

    CHECK(cudaDeviceSynchronize());

    double startTime, endTime;

    // use OpenCV to load a grayscale image
    cv::Mat grayImg = cv::imread("Santa-grayscale.jpg", cv::IMREAD_GRAYSCALE);
    if (grayImg.empty()) return -1;

    // obtain image's height, width, and number of channels
    unsigned int nRows = grayImg.rows, nCols = grayImg.cols, nChannels = grayImg.channels();

    // use OpenCV's blur
    cv::Mat blurredImg_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = CPUTimer();
    cv::blur(grayImg, blurredImg_opencv, cv::Size(2 * FILTER_RADIUS + 1, 2 * FILTER_RADIUS + 1), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    printf("openCV's blur (CPU): %f s\n\n", endTime-startTime);

    // CPU blurring
    cv Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
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
    verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);
    verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);
    verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);

    return 0;

}
