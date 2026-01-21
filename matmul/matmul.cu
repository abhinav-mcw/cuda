#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 32

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// CUDA kernel
__global__ 
void matmulGPU(float* d_A, float* d_B, float* d_C, int m, int n, int k)
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

// m*n @ n*k = m*k

void matmulCPU(float* h_A, float* h_B, float* h_C, int m, int n, int k) 
{
    for(int row=0; row<m; row++){
        for(int col=0; col<k; col++){
            float sum=0.0f;
            for (int i=0; i<n; i++){
                sum+=h_A[row*n+i]*h_B[i*k+col];
            }
            h_C[row*k+col]=sum;
        }
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    // Matrix sizes
    int m = 30, n = 40, k = 50;

    int size_A = m * n * sizeof(float);
    int size_B = n * k * sizeof(float);
    int size_C = m * k * sizeof(float);

    h_a = (float*)malloc(size_A);
    h_b = (float*)malloc(size_B);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu = (float*)malloc(size_C);

    // Host matrices (dummy data)
    init_matrix(h_a, m, n);
    init_matrix(h_b, n, k);

    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);


    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmulCPU(h_a, h_b, h_c_cpu, m, n, k);
        matmulGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
        cudaDeviceSynchronize();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();


    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmulCPU(h_a, h_b, h_c_cpu, m, n, k);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        matmulGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);

    // Print results
    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    bool flag=true;

    for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
            if(std::abs(h_c_cpu[i*k+j]-h_c_gpu[i*k+j])>0.0001){
                flag=false;
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
