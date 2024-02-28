#ifndef IMatMul_cuh
#define IMatMul_cuh

#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <chrono>

#define TILE_WIDTH 32

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

class IMatMul {
protected:
    float* m_baselineMatrixA; // baseline matrix A with size dimSize * dimSize
    float* m_baselineMatrixB; // baseline matrix B with size dimSize * dimSize
    float* m_resultMatrix;    // result matrix with size dimSize * dimSize
    int m_dimSize;            // contains n of an n*n matrix
public:
    /**
     * @brief Prepares the execution of the matrix multiplication. For example, copies the
     * contents of the baselineMatrix in an additional array
     */
    virtual void prepare() = 0;
    /**
     * @brief Runs the matrix multiplication algorithm
     */
    virtual void run() = 0;
    /**
     * @brief Ran whenever run is finished, for example to copy the contents of result to the actual
     * result matrix
     */
    virtual void post() = 0;

    IMatMul(float* baselineA, float* baselineB, float* result, int dimSize)
    : m_baselineMatrixA(baselineA)
    , m_baselineMatrixB(baselineB)
    , m_resultMatrix(result)
    , m_dimSize(dimSize)
    { }

    /**
     * @brief Compares the two results matrix from this and @param other
     * @param other the other IMatMul to compare to
     * @return true if both results are equal, else false
     */
    bool operator==(IMatMul& other) {
        int size = m_dimSize * m_dimSize;
        for (int i = 0; i < size; i++) {
            // checks against standard c++ float epsilon
            if(abs(m_resultMatrix[i] - other.m_resultMatrix[i]) > __FLT_EPSILON__) {
                return false;
            }
        }
        return true;
    }

    virtual ~IMatMul() = default;
};



#endif
