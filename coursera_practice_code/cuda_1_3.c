__global__ void PictureKernel(float* d_Pin, float* d_Pout, int n, int m)
{
    // Calculate the row # of the d_Pin and d_Pout element
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the column # of the d_Pin and d_Pout element
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread computes one element of d_Pout if in range
    if ((Row < m) && (Col < n)) {
        d_Pout[Row * n + Col] = 2.0 * d_Pin[Row * n + Col];
    }
}

// assume that the picture is m x n,
// m pixels in y dimension and n pixels in x dimension
// input d_Pin has been allocated on and copied to device
// output d_Pout has been allocated on device
...
dim3 DimGrid((n-1)/16 + 1, (m-1)/16 + 1, 1);
dim3 DimBlock(16, 16, 1);
PictureKernel<<<DimGrid, DimBlock>>>(d_Pin, d_Pout, m, n);