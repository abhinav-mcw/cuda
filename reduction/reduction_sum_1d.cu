#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 1024

// Initialize matrix with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX * 10;
        // vec[i] = 1.0f;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ 
void reductionSumGPU(float *input, float *h_c_gpu) {

    __shared__ float partialSum[2*BLOCK_SIZE];

    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    partialSum[t] = input[start+t]; // 0,1,2,3
    partialSum[blockDim.x+t]=input[start+blockDim.x+t]; // 4,5,6,7

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) 
    {
        __syncthreads();
        if (t % stride == 0) 
        {
            partialSum[2 * t] += partialSum[2 * t + stride];
        }
    }
    __syncthreads();
    *h_c_gpu = partialSum[0];
}

void reductionSumCPU(float* h_a, float* h_c_cpu, int n){
    float sum=0.0f;
    for(int i=0; i<n; i++){
        sum+=h_a[i];
    }
    *h_c_cpu=sum;
}

int main(){
    float *h_a, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_c;

    int n = 2048;

    int size_A = n * sizeof(float);
    int size_C = sizeof(float);

    h_a = (float*)malloc(size_A);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu = (float*)malloc(size_C);

    init_vector(h_a, n);

    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_c, size_C);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);

    // int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks = (n/2) / BLOCK_SIZE;

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        reductionSumCPU(h_a, h_c_cpu, n);
        reductionSumGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_c);
        cudaDeviceSynchronize();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        reductionSumCPU(h_a, h_c_cpu, n);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        reductionSumGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_c);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);

    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    bool flag;

    if(abs(*h_c_cpu-*h_c_gpu)<.01)
        flag=true;
    else{
        flag=false;
        printf("%f %f \n", *h_c_cpu, *h_c_gpu);
    }
    printf("sum equal: %d", flag);

    free(h_a);
    free(h_c_cpu);
    free(h_c_gpu);

    cudaFree(d_a);
    cudaFree(d_c);

    return 0;

}