#ifndef SharedGpuMatMul_cuh
#define SharedGpuMatMul_cuh

#include "IGpuMatMul.cuh"

class SharedGpuMatMul : public IGpuMatMul {
public:
    SharedGpuMatMul(float* baselineA, float* baselineB, float* result, int dimSize)
    :IGpuMatMul(baselineA, baselineB, result, dimSize, "Shared") {
        prepare();
        run();
        post();
    };

    virtual void run() {
        // determine block size -> if dimSize is larger than TILE_WIDTH, computer, else just use one
        int blockSize = 1;
        if(m_dimSize > TILE_WIDTH) {
            blockSize = (m_dimSize / TILE_WIDTH);
            if(m_dimSize % TILE_WIDTH != 0) {
                blockSize += 1;
            }
        }
        dim3 blocks(blockSize, blockSize);
        dim3 threads(TILE_WIDTH, TILE_WIDTH);

        // run the shared implementation
        matMulGpuShared<<<blocks, threads>>>(
            m_d_matrixA,
            m_d_matrixB,
            m_d_matrixC,
            m_dimSize
        );
    }
    virtual ~SharedGpuMatMul() {};
};
#endif
