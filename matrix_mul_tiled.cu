#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 2

__global__ void MatrixMulKernel(int m, int n, int k, float* A, float* B, float* C)
{
    // 1. & 2. Declare shared memory for the tiles of A and B
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // 3. Identify the block indices
    int bx = blockIdx.x;  int by = blockIdx.y;
    
    // 4. Identify the thread indices within the block
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 5. & 6. Calculate the global Row and Column indices for the element
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    // 7. Initialize the accumulator for the dot product
    float Cvalue = 0;
    
    // ... (The rest of the kernel logic would follow here)
    // 8. Loop over the A and B tiles required to compute the C element
    for (int t = 0; t < (n-1)/TILE_WIDTH+1; ++t) {

        // Collaborative loading of A and B tiles into shared memory
        // 9. Load an element from A into shared memory
        if (Row <m && t*TILE_WIDTH+tx<n ){
            ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH + tx];
        } else{
            ds_A[ty][tx]=0.0;
        }
        
        // 10. Load an element from B into shared memory
        if(t*TILE_WIDTH+ty<n && Col<k){
            ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)*k + Col];
        } else{
            ds_B[ty][tx]=0.0;
        }

        // 11. Wait for all threads in the block to finish loading their elements
        __syncthreads();

        // 12. & 13. Compute the partial dot product for this tile
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }

        // 14. Synchronize again before starting the next tile iteration
        __syncthreads();
    }

    // 16. Write the final computed value to the global memory of matrix C
    if (Row<m && Col<k)
        C[Row*k + Col] = Cvalue;
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
