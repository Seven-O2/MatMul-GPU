#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include <chrono>
#include <omp.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "Eigen/Dense"
/* This is our CUDA call wrapper, we will use in PAC.
*
*  Almost all CUDA calls should be wrapped with this makro.
*  Errors from these calls will be catched and printed on the console.
*  If an error appears, the program will terminate.
*
* Example: gpuErrCheck(cudaMalloc(&deviceA, N * sizeof(int)));
*          gpuErrCheck(cudaMemcpy(deviceA, hostA, N * sizeof(int), cudaMemcpyHostToDevice));
*/
#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}

#define cublasErrCheck(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "cuBLAS assert: " << code << " " << file << " " << line << std::endl;
        if (abort)
        {
            exit(code);
        }
    }
}

void populateMatrixBuffer(float* buffer1, float*buffer2, int dimSize)
{
    // Init of matrix buffer
    for (int i = 0; i < dimSize; i++) {
        for (int j = 0; j < dimSize; j++) {
            buffer1[i * dimSize + j] = 1.0f / (j + 1);
            buffer2[i * dimSize + j] = 1.0f / (j + 1);
        }
    }
}

/**
 * @brief  Compare result arrays CPU vs GPU result. If no diff, the result pass.
 * @param matrixLeft  the lhs matrix to compare
 * @param matrixRight the rhs matrix to compare
 * @param size the size of the matrices (dimSize * dimSize)
 * @return true if the matrices are equal
 */
bool compareResultVec(float* matrixLeft, float* matrixRight, int size) {
    for (int i = 0; i < size; i++) {
        // checks if only the lsb in the number is be flipped
        if(abs(matrixLeft[i] - matrixRight[i]) > __FLT_EPSILON__) {
            return false;
        }
    }
    return true;
}

/**
 * @brief This method runs a matrix multiplication on cuBLAS.
 * Thanks to https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
 * I was able to fill in the respective values to the cublasSgemm method. See
 * https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference for API doc
 * @param handle       A cudaHandler. Since cuBLAS recommends reusing the handle when multiple runs are mad
 * @param matrixA      Host Matrix A
 * @param matrixB      Host Matrix B
 * @param matrixResult Host Result matrix
 * @param dimSize      Dimension of the matrix
 * @return the time the execution took
 */
float runCublas(cublasHandle_t& handle, float* matrixA, float* matrixB, float* matrixResult, int dimSize) {
    int size = dimSize * dimSize;
    cudaEvent_t start, stop;
    gpuErrCheck(cudaEventCreate(&start));
    gpuErrCheck(cudaEventCreate(&stop));
    gpuErrCheck(cudaEventRecord(start, 0));
    // Allocate the arrays on device
    float *d_matrixA, *d_matrixB, *d_result;
    gpuErrCheck(cudaMalloc(&d_matrixA, size * sizeof(float)));
    gpuErrCheck(cudaMalloc(&d_matrixB, size * sizeof(float)));
    gpuErrCheck(cudaMalloc(&d_result, size * sizeof(float)));

    // Copy data from host do device
    gpuErrCheck(cudaMemcpy(d_matrixA, matrixA, size * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(d_matrixB, matrixB, size * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1;
    const float beta = 0;
    cublasErrCheck(
        cublasSgemm(handle,           // handle to the cuBLAS library context
                    CUBLAS_OP_N,      // no transpose, even though the original matrices are row-major and cublas uses col-major
                    CUBLAS_OP_N,      //                                  "
                    dimSize,          // number of rows in matrix A and Result
                    dimSize,          // number of columns in Matrix B and Result
                    dimSize,          // number of rows in matrix A and columns in Matrix B
                    &alpha,           // scalar for multiplication
                    d_matrixA,        // Matrix A
                    dimSize,          // leading dimension of two-dimensional array used to store each matrix A[i].
                    d_matrixB,        // Matrix B
                    dimSize,          // leading dimension of two-dimensional array used to store each matrix B[i].
                    &beta,            // Scalar when A*B +C, where C is 0 when only multiplication is needed
                    d_result,         // Result Matrix
                    dimSize)          // Leading dimension of a two dimensional array use to store the matrix Result[i]
    );

    // Copy results to host memory
    gpuErrCheck(cudaMemcpy(matrixResult, d_result, size * sizeof(float), cudaMemcpyDeviceToHost));

    //Free GPU memory
    gpuErrCheck(cudaFree(d_matrixA));
    gpuErrCheck(cudaFree(d_matrixB));
    gpuErrCheck(cudaFree(d_result));
    gpuErrCheck(cudaEventRecord(stop, 0));
    gpuErrCheck(cudaEventSynchronize(stop));

    float executionTime = 0.0;
    gpuErrCheck(cudaEventElapsedTime(&executionTime, start, stop));
    return executionTime;
}

using Eigen::MatrixXf;
/**
 * @brief This method runs a matrix multiplication on eigen
 * @param matrixA      Baseline Matrix A
 * @param matrixB      Baseline Matrix B
 * @param matrixResult Result matrix
 * @param dimSize      Dimensions of the matrix
 */
void runEigen(float* matrixA, float* matrixB, float* matrixResult, int dimSize){
    // create eigen matrix but using the already allocated and initialized arrays
    Eigen::Map<MatrixXf> eigenMatrixA(matrixA, dimSize, dimSize);
    Eigen::Map<MatrixXf> eigenMatrixB(matrixB, dimSize, dimSize);
    Eigen::Map<MatrixXf> eigenResult(matrixResult, dimSize, dimSize);

    // Calculate
    eigenResult = eigenMatrixA * eigenMatrixB;
}

/**
 * @brief Runs a check on both eigen and cublas implementationto check if they return the same as
 * the naive cpu version
 * @return true if both implementations return the same as the naive CPU version
 */
bool runCheck(cublasHandle_t& handle) {
    int DIM_SIZE = 32;
    float* matrixA = new float[DIM_SIZE * DIM_SIZE];
    float* matrixB = new float[DIM_SIZE * DIM_SIZE];
    populateMatrixBuffer(matrixA, matrixB, DIM_SIZE);

    float* eigenResult = new float[DIM_SIZE * DIM_SIZE];
    runEigen(matrixA, matrixB, eigenResult, DIM_SIZE);

    float* cuBLASresult = new float[DIM_SIZE * DIM_SIZE];
    runCublas(handle, matrixA, matrixB, cuBLASresult, DIM_SIZE);

    float* naiveResult = new float[DIM_SIZE * DIM_SIZE];
    // Naive CPU implementation
    float sum;
    for (int i = 0; i < DIM_SIZE; i++) {
        for (int j = 0; j < DIM_SIZE; j++) {
            sum = 0.0;
            for (int n = 0; n < DIM_SIZE; n++){
                sum += matrixA[i * DIM_SIZE + n] * matrixB[n * DIM_SIZE + j];
            }
            naiveResult[i * DIM_SIZE + j] = sum;
        }
    }

    // Equality check
    bool areEqual = true;
    areEqual &= compareResultVec(naiveResult, eigenResult, DIM_SIZE * DIM_SIZE);
    areEqual &= compareResultVec(naiveResult, cuBLASresult, DIM_SIZE * DIM_SIZE);
    return areEqual;
}

void runBenchmark(cublasHandle_t& handle, int start, int stop, int stride, bool runCPU) {
    for(int DIM_SIZE = start; DIM_SIZE <= stop; DIM_SIZE+= stride){
        float* matrixA = new float[DIM_SIZE * DIM_SIZE];
        float* matrixB = new float[DIM_SIZE * DIM_SIZE];
        populateMatrixBuffer(matrixA, matrixB, DIM_SIZE);

        // Run cuBLAS
        float* cuBLASresult = new float[DIM_SIZE * DIM_SIZE];
        float cublasExecTime = runCublas(handle, matrixA, matrixB, cuBLASresult, DIM_SIZE);

        if(runCPU) {
            // Run Eigen
            auto startTimeEigen = std::chrono::high_resolution_clock::now();
            float* eigenResult = new float[DIM_SIZE * DIM_SIZE];
            runEigen(matrixA, matrixB, eigenResult, DIM_SIZE);
            auto endTimeEigen = std::chrono::high_resolution_clock::now();

            // Output
            auto durationEigen = std::chrono::duration_cast<std::chrono::microseconds>(endTimeEigen - startTimeEigen).count();
            std::cout << DIM_SIZE << "   " <<(double)cublasExecTime << "   " << (double)durationEigen / 1000.0 <<std::endl;

            delete[] eigenResult;
        } else {
            std::cout << DIM_SIZE << "   " <<(double)cublasExecTime << std::endl;
        }

        delete[] cuBLASresult;
        delete[] matrixA;
        delete[] matrixB;
    }
}

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    std::cout << "RUNNING CHECK... ";
    if(runCheck(handle)) {
        std::cout << "OK" << std::endl;
    } else {
        std::cout << "FAILED!" << std::endl;
        return 1;
    }

    // Warmup
    runBenchmark(handle, 4096, 4096, 1, true);

    runBenchmark(handle, 1, 5001, 10, true);
    runBenchmark(handle, 5000, 10000, 1000, true);
    runBenchmark(handle, 10000, 26000, 1000, false);


    cublasDestroy(handle);

    return 0;
}
