#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 3

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        // mat[i] = (float)rand() / RAND_MAX;
        mat[i] = 1.0f;
    }
}

// CUDA kernel for 1 channel (2d conv)
__global__ void conv2dGPU(float *d_A, float *d_B, float *d_C, int height, int width, int mask_width, int output_height, int output_width, int stride)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    float output = 0.0f;

    if (Row < output_height && Col < output_width)
    {
        for (int i = 0; i < mask_width; i++)
        {
            for (int j = 0; j < mask_width; j++)
            {
                output += d_A[(i + Row * stride) * width + (Col * stride + j)] * d_B[i * mask_width + j];
            }
        }
        d_C[Row * output_width + Col] = output;
    }

    __syncthreads();
}

void conv2dCPU(float *h_A, float *h_B, float *h_C, int height, int width, int mask_width, int output_height, int output_width, int stride)
{
    for (int row = 0; row < output_height; row++)
    {
        for (int col = 0; col < output_width; col++)
        {
            float sum = 0.0f;
            for (int i = 0; i < mask_width; i++)
            {
                for (int j = 0; j < mask_width; j++)
                {
                    sum += h_A[(row * stride + i) * width + (col * stride + j)] * h_B[i * mask_width + j];
                }
            }
            h_C[row * output_width + col] = sum;
        }
    }
}

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    // Matrix sizes
    int height = 50, width = 50;
    int mask_width = 2;
    int stride = 2;

    int output_height = (height - mask_width) / stride + 1;
    int output_width = (width - mask_width) / stride + 1;

    int size_A = height * width * sizeof(float);
    int size_B = mask_width * mask_width * sizeof(float);
    int size_C = output_height * output_width * sizeof(float);

    h_a = (float *)malloc(size_A);
    h_b = (float *)malloc(size_B);
    h_c_cpu = (float *)malloc(size_C);
    h_c_gpu = (float *)malloc(size_C);

    // Host matrices (dummy data)
    init_matrix(h_a, height, width);
    init_matrix(h_b, mask_width, mask_width);

    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((output_width - 1) / BLOCK_SIZE + 1, (output_height - 1) / BLOCK_SIZE + 1);

    printf("output height %d \n", output_height);
    printf("output width %d \n", output_width);

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 1; i++)
    {
        conv2dCPU(h_a, h_b, h_c_cpu, height, width, mask_width, output_height, output_width, stride);
        conv2dGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, height, width, mask_width, output_height, output_width, stride);
        cudaDeviceSynchronize();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        conv2dCPU(h_a, h_b, h_c_cpu, height, width, mask_width, output_height, output_width, stride);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        conv2dGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, height, width, mask_width, output_height, output_width, stride);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    bool flag = true;

    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            if (std::abs(h_c_cpu[i * output_width + j] - h_c_gpu[i * output_width + j]) > .01)
            {
                flag = false;
                break;
            }
        }
    }

    printf("matrix equal: %d", flag);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
