#ifndef IGpuMatMul_cuh
#define IGpuMatMul_cuh

#include "IMatMul.cuh"
class IGpuMatMul : public IMatMul {
protected:
    float* m_d_matrixA; // device matrix A (baseline A)
    float* m_d_matrixB; // device matrix B (baseline B)
    float* m_d_matrixC; // device matrix C (result)
    cudaEvent_t m_calculationStartEvent; // Cuda event that keeps the start point of the algorithm's start
    cudaEvent_t m_calculationStopEvent;  // Cuda event that keeps the stop point of the algorithm's stop
    cudaEvent_t m_loadingStartEvent;     // Cuda event that keeps the start point of the loading (D2H and H2D) operations
    cudaEvent_t m_loadingStopEvent;      // Cuda event that keeps the stop point of the loading (D2H and H2D) operations
    int m_memorySize;
    string m_timeReportPrefix;
public:
    IGpuMatMul(float* baselineA, float* baselineB, float* result, int dimSize, string timeReportPrefix)
    :IMatMul(baselineA, baselineB, result, dimSize)
    , m_timeReportPrefix(timeReportPrefix)
    {}
    /**
     * @brief Copy the contents of the baseline matrices to the CUDA device. Also measures the time
     * this took.
     */
    virtual void prepare() {
        m_memorySize = m_dimSize * m_dimSize * sizeof(float);

        // Load data and stop time of this
        gpuErrCheck(cudaEventCreate(&m_loadingStartEvent));
        gpuErrCheck(cudaEventCreate(&m_loadingStopEvent));
        gpuErrCheck(cudaEventRecord(m_loadingStartEvent, 0));
        gpuErrCheck(cudaMalloc(&m_d_matrixA, m_memorySize));
        gpuErrCheck(cudaMalloc(&m_d_matrixB, m_memorySize));
        gpuErrCheck(cudaMalloc(&m_d_matrixC, m_memorySize));
        gpuErrCheck(cudaMemcpy(m_d_matrixA, m_baselineMatrixA, m_memorySize, cudaMemcpyHostToDevice));
        gpuErrCheck(cudaMemcpy(m_d_matrixB, m_baselineMatrixB, m_memorySize, cudaMemcpyHostToDevice));
        gpuErrCheck(cudaEventRecord(m_loadingStopEvent, 0));
        gpuErrCheck(cudaEventSynchronize(m_loadingStopEvent));

        // Start calculation event
        gpuErrCheck(cudaEventCreate(&m_calculationStartEvent));
        gpuErrCheck(cudaEventCreate(&m_calculationStopEvent));
        gpuErrCheck(cudaEventRecord(m_calculationStartEvent, 0));
    }

    virtual void run() = 0;

    virtual void post() {
        float executionTime;
        float h2dloadingTime;
        float d2hloadingTime;

        // Finish recording of caluclation event and calculate exection time
        gpuErrCheck(cudaEventRecord(m_calculationStopEvent, 0));
        gpuErrCheck(cudaEventSynchronize(m_calculationStopEvent));
        gpuErrCheck(cudaEventElapsedTime(&executionTime, m_calculationStartEvent, m_calculationStopEvent));
        gpuErrCheck(cudaPeekAtLastError());

        // Calculate h2d loading time
        gpuErrCheck(cudaEventElapsedTime(&h2dloadingTime, m_loadingStartEvent, m_loadingStopEvent));

        // Record and calculate d2h loading time and time to free memory
        gpuErrCheck(cudaEventRecord(m_loadingStartEvent, 0));
        gpuErrCheck(cudaMemcpy(m_resultMatrix, m_d_matrixC, m_memorySize, cudaMemcpyDeviceToHost));
        gpuErrCheck(cudaFree(m_d_matrixA));
        gpuErrCheck(cudaFree(m_d_matrixB));
        gpuErrCheck(cudaFree(m_d_matrixC));
        gpuErrCheck(cudaEventRecord(m_loadingStopEvent, 0));
        gpuErrCheck(cudaEventSynchronize(m_loadingStopEvent));
        gpuErrCheck(cudaEventElapsedTime(&d2hloadingTime, m_loadingStartEvent, m_loadingStopEvent));

        std::cout << m_timeReportPrefix << " GPU time [ms]: " << executionTime << " " << h2dloadingTime + d2hloadingTime << std::endl;
    }
};
#endif
