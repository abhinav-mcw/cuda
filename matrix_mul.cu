#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16

// CUDA kernel
__global__ void MatrixMulKernel(int m, int n, int k,
                                float* d_A, float* d_B, float* d_C)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < m && Col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += d_A[Row * n + i] * d_B[i * k + Col];
        }
        d_C[Row * k + Col] = sum;
    }
}

void matrixMul(float* h_A, float* h_B, float* h_C, int m, int n, int k){
    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * k * sizeof(float);
    size_t sizeC = m * k * sizeof(float);

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(
        (k - 1) / TILE_WIDTH +1,
        (m - 1) / TILE_WIDTH +1,
        1
    );

    // Kernel launch
    MatrixMulKernel<<<dimGrid, dimBlock>>>(m, n, k, d_A, d_B, d_C);
    // cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    // Matrix sizes
    int m = 4, n = 3, k = 5;

    // Host matrices (dummy data)
    float h_A[12] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };

    float h_B[15] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    };

    float h_C[20] = {0};

    matrixMul(h_A, h_B, h_C, m, n, k);

    // Print result
    std::cout << "Result matrix C (" << m << " x " << k << "):\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            std::cout << h_C[i * k + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
