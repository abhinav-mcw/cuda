__global__ 
void MatrixMulKernel(int m, int n, int k, float* A, float* B, float* C)
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