#include <cuda_runtime.h>
#include <iostream>

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_C, size);

    // dim3 DimBlock(256, 1, 1);
    // dim3 DimGrid((n + DimBlock.x - 1) / DimBlock.x, 1, 1);
    // vecAddKernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n);
    
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch error: "
                << cudaGetErrorString(err) << std::endl;
    }

    // cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main()
{
    const int n = 10;
    float h_A[n], h_B[n], h_C[n];

    // Dummy input initialization
    for (int i = 0; i < n; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Call CUDA vector addition
    vecAdd(h_A, h_B, h_C, n);

    // Print result
    std::cout << "Result:\n";
    for (int i = 0; i < n; i++)
    {
        std::cout << h_A[i] << " + " << h_B[i]
                  << " = " << h_C[i] << std::endl;
    }

    return 0;
}
