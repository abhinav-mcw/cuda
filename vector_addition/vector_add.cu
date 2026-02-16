#include <cuda_runtime.h>
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void vecAddGPU(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vecAddCPU(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    const int n = 4096;
    float h_a[n], h_b[n], h_c_cpu[n], h_c_gpu[n];

    // Dummy input initialization
    for (int i = 0; i < n; i++)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    int size = n * sizeof(float);
    float *d_a, *d_b, *d_c;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc((void **)&d_a, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_b, size);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_c, size);

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++)
    {
        vecAddCPU(h_a, h_b, h_c_cpu, n);
        vecAddGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vecAddCPU(h_a, h_b, h_c_cpu, n);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        vecAddGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, n);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
