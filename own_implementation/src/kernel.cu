#include <iostream>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MatMul/IMatMul.cuh"
#include "MatMul/CpuMatMul.cuh"

using namespace std;

#define CPU_MAX_DIMENSION 1024
#define GPU_NAIVE_MAX_DIMENSION 2048
#define GPU_SHARED_MAX_DIMENSION 4096

/* Helper which populates a matrix buffer (dimSize*dimSize).
*
* Think of this as it would load the data from disk or somewhere else.
* This dummy data is only used to fill the buffer as fast as possible.
*/
void populateMatrixBuffer(float* buffer, int dimSize)
{
    // Init of matrix buffer
    for (int i = 0; i < dimSize; i++) {
        for (int j = 0; j < dimSize; j++) {
            buffer[i * dimSize + j] = 1.0f / (j + 1);
        }
    }
}

__global__ void matMulGpuShared(float* d_matrixA, float* d_matrixB, float* d_matrixC, int dimSize) {
    __shared__ float matrixAshared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float matrixBshared[TILE_WIDTH][TILE_WIDTH];
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int idx = 0;

    float sum = 0;
    // calculate
    for(size_t nTile = 0; nTile < gridDim.x; ++nTile) {

        // Matrix A
        idx = row * dimSize + nTile * TILE_WIDTH + threadIdx.x;
        if(idx >= dimSize * dimSize) {
            matrixAshared[threadIdx.y][threadIdx.x] = 0; // padded
        } else {
            matrixAshared[threadIdx.y][threadIdx.x] = d_matrixA[idx];
        }

        // Matrix B
        idx = col + (nTile * TILE_WIDTH + threadIdx.y) * dimSize;
        if(idx >= dimSize * dimSize) {
            matrixBshared[threadIdx.y][threadIdx.x] = 0; // padded
        } else {
            matrixBshared[threadIdx.y][threadIdx.x] = d_matrixB[idx];
        }
        __syncthreads();

        for(int tileElement = 0; tileElement < TILE_WIDTH; ++tileElement) {
            sum +=  matrixAshared[threadIdx.y][tileElement] * matrixBshared[tileElement][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < dimSize && col < dimSize) {
        d_matrixC[row * dimSize + col] = sum;
    }
}

__global__ void matMulGpuNaive(float* d_matrixA, float* d_matrixB, float* d_matrixC, int dimSize) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row < dimSize && col < dimSize){
        float sum = 0;
        for(int i = 0; i < dimSize; i++) {
            sum += d_matrixA[row * dimSize + i] * d_matrixB[i * dimSize + col];
        }
        d_matrixC[row * dimSize + col] = sum;
    }
}

/*
 * This is ugly, I'm aware. But this way we do not need another file containing the kernels
 * Cuda does not allow __global__ functions inside classes, and they have to be definied inside a
 * .cu file
 */
#include "MatMul/NaiveGpuMatMul.cuh"
#include "MatMul/SharedGpuMatMul.cuh"

void runBenchmark(int DIM_SIZE) {
    cout << "==================================> DIM_SIZE: " << DIM_SIZE << endl;
    float* matrixA = new float[DIM_SIZE * DIM_SIZE];
    float* matrixB = new float[DIM_SIZE * DIM_SIZE];
    populateMatrixBuffer(matrixA, DIM_SIZE);
    populateMatrixBuffer(matrixB, DIM_SIZE);

    // GPU shared
    float* sharedGpuResult = new float[DIM_SIZE * DIM_SIZE];
    SharedGpuMatMul sharedGpuMatMul(matrixA, matrixB, sharedGpuResult, DIM_SIZE);

    // If GPU Naive finishes in a useable time
    if(DIM_SIZE <= GPU_NAIVE_MAX_DIMENSION) {
        // GPU Shared
        float* naiveGpuResult = new float[DIM_SIZE * DIM_SIZE];
        NaiveGpuMatMul naiveGpuMatMul(matrixA, matrixB, naiveGpuResult, DIM_SIZE);
        delete[] naiveGpuResult;
    }


    // If CPU finishes in a useable time
    if(DIM_SIZE <= CPU_MAX_DIMENSION) {
        // CPU
        float* cpuResult = new float[DIM_SIZE * DIM_SIZE];
        CpuMatMul cpuMatMul(matrixA, matrixB, cpuResult, DIM_SIZE);
        delete[] cpuResult;
    }

    delete[] matrixA;
    delete[] matrixB;
    delete[] sharedGpuResult;
}

bool runBenchmarkWithCompare(int DIM_SIZE) {
    std::cout << "================> DIM_SIZE: " << DIM_SIZE << std::endl;
    float* matrixA = new float[DIM_SIZE * DIM_SIZE];
    float* matrixB = new float[DIM_SIZE * DIM_SIZE];
    populateMatrixBuffer(matrixA, DIM_SIZE);
    populateMatrixBuffer(matrixB, DIM_SIZE);
    bool resultsEqual = true;

    // GPU Shared
    float* sharedGpuResult = new float[DIM_SIZE * DIM_SIZE];
    SharedGpuMatMul sharedGpuMatMul(matrixA, matrixB, sharedGpuResult, DIM_SIZE);

    // GPU Naive
    float* naiveGpuResult = new float[DIM_SIZE * DIM_SIZE];
    NaiveGpuMatMul naiveGpuMatMul(matrixA, matrixB, naiveGpuResult, DIM_SIZE);

    float* cpuResult = new float[DIM_SIZE * DIM_SIZE];
    CpuMatMul cpuMatMul(matrixA, matrixB, cpuResult, DIM_SIZE);

    resultsEqual &= (naiveGpuMatMul == sharedGpuMatMul);
    resultsEqual &= naiveGpuMatMul == cpuMatMul;
    resultsEqual &= sharedGpuMatMul == cpuMatMul;

    delete[] matrixA;
    delete[] matrixB;
    delete[] sharedGpuResult;
    delete[] naiveGpuResult;
    delete[] cpuResult;

    return resultsEqual;
}

int main() {

    // proof that the solution works
    std::cout << "===================== PROOF =====================" <<std::endl;
    srand(time(NULL));
    for(int i = 0; i < 5; i++){
        if(runBenchmarkWithCompare(rand() % CPU_MAX_DIMENSION + 1)){
            std::cout << "Passed!" <<std::endl;
        } else {
            std::cout << "Failed!" <<std::endl;
            return 1;
        }
    }

    // warmup
    std::cout << "===================== WARMUP =====================" << std::endl;
    runBenchmark(CPU_MAX_DIMENSION);  // Warms up with max matrix size the CPU handles
    runBenchmark(GPU_NAIVE_MAX_DIMENSION); // Warms up with max matrix size the Naive GPU implementation handles
    runBenchmark(GPU_SHARED_MAX_DIMENSION); // Warms up with max matrix size the Shared Memory implementation handles


    std::cout << "===================== Nums 1 - 1000, spaced 10 =====================" <<std::endl;
    for(int DIM_SIZE = 1; DIM_SIZE <= 1000; DIM_SIZE += 10) {
        runBenchmark(DIM_SIZE);
    }

    std::cout << "===================== Nums 1000 - 4000, spaced 100 =====================" <<std::endl;
    for(int DIM_SIZE = 1000; DIM_SIZE <= 4000; DIM_SIZE += 100) {
        runBenchmark(DIM_SIZE);
    }

    std::cout << "===================== Nums divideable by 32 =====================" <<std::endl;
    for (int DIM_SIZE = 32; DIM_SIZE <= GPU_SHARED_MAX_DIMENSION; DIM_SIZE <<= 1) {
        runBenchmark(DIM_SIZE);
    }

    return 0;
}
