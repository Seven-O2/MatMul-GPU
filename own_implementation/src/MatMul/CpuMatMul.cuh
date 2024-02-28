#ifndef CpuMatMul_cuh
#define CpuMatMul_cuh

#include "IMatMul.cuh"
class CpuMatMul : public IMatMul {
public:
    CpuMatMul(float* baselineA, float* baselineB, float* result, int dimSize)
    :IMatMul(baselineA, baselineB, result, dimSize) {
        prepare();
        run();
        post();
    };

    virtual void prepare() { /* No operation needed */ }
    virtual void post() { /* No operation needed */ }
    virtual void run() {
        auto startTime = std::chrono::high_resolution_clock::now();

        // naive CPU algorithm
        float sum;
        for (int i = 0; i < m_dimSize; i++) {
            for (int j = 0; j < m_dimSize; j++) {
                sum = 0.0;
                for (int n = 0; n < m_dimSize; n++) {
                    sum += m_baselineMatrixA[i * m_dimSize + n] * m_baselineMatrixB[n * m_dimSize + j];
                }
                m_resultMatrix[i * m_dimSize + j] = sum;
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        std::cout << "CPU time [ms]: " << (double)duration / 1000.0 << std::endl;
    }
};
#endif
