#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

// helper function to transpose a matrix
std::vector<data_type> transpose_matrix(const data_type *A, int m, int n) {
    std::vector<data_type> result(m * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result[j * m + i] = A[i * n + j];
        }
    }
    return result;
}

int main(int argc, char *argv[]) {
    printf("Approach 1:\n");
    //printf("Approach 2:\n");
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = 3;   // A is of size m by k
    const int n = 4;   // B is of size k by n
    const int k = 2;

    /*
     *   A = | 1.0 | 2.0 |
     *       | 3.0 | 4.0 |
    *        | 5.0 | 6.0 |
     *
     *   B = | 7.0 | 8.0 | 9.0 | 10.0 |
     *       | 11.0 | 12.0 | 13.0 | 14.0 |
     */

    // *********** Suppose data of A and B are stored in row-major order
    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};   
    const std::vector<data_type> B = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};
    std::vector<data_type> C(m * n);
    const data_type alpha = 1.0;    // C = alpha * A * B + beta * C
    const data_type beta = 0.0;

    //Complete your code here
    //approach 1: transposing the row ordered matrices and computing C = AB
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const std::vector<data_type> A_T = transpose_matrix(A.data(), m, k);
    const std::vector<data_type> B_T = transpose_matrix(B.data(), k, n);
    
    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");
    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
 
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A_T.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(data_type) * B_T.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A_T.data(), sizeof(data_type) * A_T.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B_T.data(), sizeof(data_type) * B_T.size(), cudaMemcpyHostToDevice, stream));


    /* step 3: compute */
    CUBLAS_CHECK(cublasGemmEx(
        cublasH, transa, transb, m, n, k, &alpha, 
        d_A, traits<data_type>::cuda_data_type, lda, 
        d_B, traits<data_type>::cuda_data_type, ldb, 
        &beta, d_C, traits<data_type>::cuda_data_type, ldc,
        CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
                               stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // //approach 2: C^T = (AB)^T = B^T A^T
    // const int lda = m;
    // const int ldb = k;
    // const int ldc = m;

    // const std::vector<data_type> A_T = transpose_matrix(A.data(), m, k);
    // const std::vector<data_type> B_T = transpose_matrix(B.data(), k, n);

    // printf("A\n");
    // print_matrix(m, k, A.data(), lda);
    // printf("=====\n");
    // printf("B\n");
    // print_matrix(k, n, B.data(), ldb);
    // printf("=====\n");

    // std::vector<data_type> C_T(n * m);

    // const int lda2 = k; 
    // const int ldb2 = n;
    // const int ldc2 = n; 

    // data_type *d_AT = nullptr;
    // data_type *d_BT = nullptr; 
    // data_type *d_CT = nullptr;

    // CUBLAS_CHECK(cublasCreate(&cublasH));
    // CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // CUDA_CHECK(cudaMalloc((void**)&d_AT, sizeof(data_type) * A_T.size()));
    // CUDA_CHECK(cudaMalloc((void**)&d_BT, sizeof(data_type) * B_T.size()));
    // CUDA_CHECK(cudaMalloc((void**)&d_CT, sizeof(data_type) * C_T.size()));

    // CUDA_CHECK(cudaMemcpyAsync(d_AT, A_T.data(), sizeof(data_type) * A_T.size(), cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_BT, B_T.data(), sizeof(data_type) * B_T.size(), cudaMemcpyHostToDevice, stream));  

    // // C_T = B_T * A_T = (AB)^T
    // CUBLAS_CHECK(cublasGemmEx(
    //     cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
    //     n, m, k, &alpha,
    //     d_BT, traits<data_type>::cuda_data_type, ldb2,
    //     d_AT, traits<data_type>::cuda_data_type, lda2,
    //     &beta,
    //     d_CT, traits<data_type>::cuda_data_type, ldc2,
    //     CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // CUDA_CHECK(cudaMemcpyAsync(C_T.data(), d_CT, sizeof(data_type) * C_T.size(), cudaMemcpyDeviceToHost, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));


    // Print the result matrix C
    /*
     *   C = |  29.0   |   32.0   |   35.0   |   38.0 | 
             |  65.0   |   72.0   |   79.0   |   86.0 | 
             | 101.0   |  112.0   |  123.0   |  134.0 | 
     */

    printf("C\n");
    print_matrix(m, n, C.data(), 3);
    // //printf("C_T\n");
    print_matrix(n, m, C_T.data(), n);
    printf("=====\n");

    /* free resources */
    // CUDA_CHECK(cudaFree(d_A));
    // CUDA_CHECK(cudaFree(d_B));
    // CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_AT));
    CUDA_CHECK(cudaFree(d_BT));
    CUDA_CHECK(cudaFree(d_CT));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
