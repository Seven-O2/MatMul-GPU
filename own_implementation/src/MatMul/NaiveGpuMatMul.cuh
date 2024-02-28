#ifndef NaiveGpuMatMul_cuh
#define NaiveGpuMatMul_cuh

#include "IGpuMatMul.cuh"

class NaiveGpuMatMul : public IGpuMatMul {
public:
    NaiveGpuMatMul(float* baselineA, float* baselineB, float* result, int dimSize)
    :IGpuMatMul(baselineA, baselineB, result, dimSize, "Naive") {
        prepare();
        run();
        post();
    };

    virtual void run() {
        // Start the naive implementation with dimSize * dimSize blocks (one thread per result)
        matMulGpuNaive<<<dim3(m_dimSize, m_dimSize), dim3(1,1)>>>(
            m_d_matrixA,
            m_d_matrixB,
            m_d_matrixC,
            m_dimSize
        );
    }

    virtual ~NaiveGpuMatMul() {};
};

#endif
