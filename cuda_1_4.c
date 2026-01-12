void matrixMulOnHost(int m, int n, int k, float* h_A, float* h_B, float* h_C) {
    for (int row=0; row<m; row++){
        for (int col=0;; col<k; k++){
            float sum=0;
            for (int i=0; i<n; i++){
                float a = h_A[row*n + i];
                float b = h_B[col+i*k];
            }
            h_C[row*k + col] = sum;
        }
    }
}

dim3 dimGrid((k-1)/TILE_WIDTH+1, (m-1)/TILE_WIDTH+1, 1);
dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

// Launch the device computation threads!
MatrixMulKernel<<<dimGrid, dimBlock>>>(m, n, k, 
                                       d_A, d_B, d_C);

__global__ void MatrixMulKernel(int m, int n, int k,
                                 float* d_A, float* d_B, float* d_C) 
{
    // Calculate the row and column of the d_C element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadidx.x;

    if ((Row < m) && (Col<k)){
        float sum=0.0f;
        for (int i=0; i<n; i++){
            sum += d_A[Row*n + i] * d_B[Col + i*k];
        }
        d_C[Row*k+Col] = sum;
    }
}