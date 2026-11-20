#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLK 16

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(e)); \
        exit(1); \
    } \
} while (0)

// 1. UNCOALESCED MATRIX TRANSPOSE
__global__ void transposeNaive(const float *in, float *out, int N) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];
}

// 2. COALESCED MATRIX TRANSPOSE
__global__ void transposeCoalesced(const float *in, float *out, int N) {

    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

// 3. NAIVE MATRIX MULTIPLICATION
__global__ void matMulNaive(const float *A, const float *B,
                            float *C, int N) {

    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    if (row < N && col < N) {

        float sum = 0.0f;

        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];

        C[row * N + col] = sum;
    }
}

// TRANSPOSE TIMING FUNCTION
float timeTranspose(bool coalesced, dim3 grid, dim3 block,
                    const float *in, float *out, int N, int reps) {

    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warm-up
    if (coalesced)
        transposeCoalesced<<<grid, block>>>(in, out, N);
    else
        transposeNaive<<<grid, block>>>(in, out, N);

    CHECK(cudaDeviceSynchronize());

    // Timing
    CHECK(cudaEventRecord(start));

    for (int r = 0; r < reps; r++) {

        if (coalesced)
            transposeCoalesced<<<grid, block>>>(in, out, N);
        else
            transposeNaive<<<grid, block>>>(in, out, N);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;

    CHECK(cudaEventElapsedTime(&ms, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms / reps;
}

// VERIFY TRANSPOSE
bool checkTranspose(const float *in, const float *out, int N) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            if (out[j * N + i] != in[i * N + j])
                return false;
        }
    }

    return true;
}

// VERIFY MATRIX MULTIPLICATION
bool checkMatMul(const float *C, int N, float expected) {

    for (int i = 0; i < N * N; i++) {

        if (fabsf(C[i] - expected) > 0.001f)
            return false;
    }

    return true;
}

int main() {

    int NT = 4096;
    int NM = 1024;

    cudaDeviceProp prop;

    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);

    // =========================================
    // PART A: MATRIX TRANSPOSE
    // =========================================

    size_t bytesT = (size_t)NT * NT * sizeof(float);

    float *h_in = (float*)malloc(bytesT);
    float *h_out = (float*)malloc(bytesT);

    for (size_t i = 0; i < (size_t)NT * NT; i++)
        h_in[i] = (float)(i % 1000);

    float *d_in, *d_out;

    CHECK(cudaMalloc(&d_in, bytesT));
    CHECK(cudaMalloc(&d_out, bytesT));

    CHECK(cudaMemcpy(d_in, h_in, bytesT,
                    cudaMemcpyHostToDevice));

    dim3 blockT(TILE, TILE);

    dim3 gridT((NT + TILE - 1) / TILE,
               (NT + TILE - 1) / TILE);

    // UNCOALESCED
    float tN = timeTranspose(false, gridT, blockT,
                             d_in, d_out, NT, 20);

    CHECK(cudaMemcpy(h_out, d_out, bytesT,
                    cudaMemcpyDeviceToHost));

    bool okN = checkTranspose(h_in, h_out, NT);

    // COALESCED
    float tC = timeTranspose(true, gridT, blockT,
                             d_in, d_out, NT, 20);

    CHECK(cudaMemcpy(h_out, d_out, bytesT,
                    cudaMemcpyDeviceToHost));

    bool okC = checkTranspose(h_in, h_out, NT);

    double gbN = 2.0 * bytesT / (tN * 1e6);
    double gbC = 2.0 * bytesT / (tC * 1e6);

    printf("\n=== PART A: MATRIX TRANSPOSE ===\n");

    printf("Matrix size: %d x %d\n", NT, NT);

    printf("Uncoalesced: %.3f ms | %.1f GB/s | %s\n",
           tN, gbN, okN ? "PASS" : "FAIL");

    printf("Coalesced:   %.3f ms | %.1f GB/s | %s\n",
           tC, gbC, okC ? "PASS" : "FAIL");

    printf("Speedup: %.2fx\n", tN / tC);

    // =========================================
    // PART B: MATRIX MULTIPLICATION
    // =========================================

    size_t bytesM = (size_t)NM * NM * sizeof(float);

    float *hA = (float*)malloc(bytesM);
    float *hB = (float*)malloc(bytesM);
    float *hC = (float*)malloc(bytesM);

    // Initialize matrices
    for (size_t i = 0; i < (size_t)NM * NM; i++) {

        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    float *dA, *dB, *dC;

    CHECK(cudaMalloc(&dA, bytesM));
    CHECK(cudaMalloc(&dB, bytesM));
    CHECK(cudaMalloc(&dC, bytesM));

    CHECK(cudaMemcpy(dA, hA, bytesM,
                    cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dB, hB, bytesM,
                    cudaMemcpyHostToDevice));

    dim3 block(BLK, BLK);

    dim3 grid((NM + BLK - 1) / BLK,
              (NM + BLK - 1) / BLK);

    // Warm-up
    matMulNaive<<<grid, block>>>(dA, dB, dC, NM);

    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;

    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    int reps = 5;

    CHECK(cudaEventRecord(start));

    for (int r = 0; r < reps; r++)
        matMulNaive<<<grid, block>>>(dA, dB, dC, NM);

    CHECK(cudaEventRecord(stop));

    CHECK(cudaEventSynchronize(stop));

    float ms;

    CHECK(cudaEventElapsedTime(&ms, start, stop));

    float tM = ms / reps;

    CHECK(cudaMemcpy(hC, dC, bytesM,
                    cudaMemcpyDeviceToHost));

    bool okM = checkMatMul(hC, NM, (float)NM);

    double gflops = 2.0 * NM * NM * (double)NM
                    / (tM * 1e6);

    printf("\n=== PART B: MATRIX MULTIPLICATION ===\n");

    printf("Matrix size: %d x %d\n", NM, NM);

    printf("Time: %.3f ms\n", tM);

    printf("Performance: %.1f GFLOPS\n", gflops);

    printf("Verification: %s\n", okM ? "PASS" : "FAIL");

    printf("C[0][0] = %.1f (Expected: %d)\n",
           hC[0], NM);

    // =========================================
    // SUMMARY
    // =========================================

    printf("\n=== SUMMARY ===\n");

    printf("Uncoalesced transpose: %.3f ms\n", tN);

    printf("Coalesced transpose:   %.3f ms\n", tC);

    printf("Naive multiplication:  %.3f ms\n", tM);

    // FREE MEMORY

    cudaFree(d_in);
    cudaFree(d_out);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(h_in);
    free(h_out);

    free(hA);
    free(hB);
    free(hC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}