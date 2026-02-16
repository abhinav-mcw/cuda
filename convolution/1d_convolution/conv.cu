#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 256

// Initialize matrix with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX * 10;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ 
void convGPU(float *N, float *M, float *P, int Mask_Width, int Width) {

    // Calculate the global index for the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;
    // Calculate the starting point in the input array N for the current mask application
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++) {
        // Boundary condition handling: check if the index is within [0, Width)
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }

    // Write the result to the output array P
    P[i] = Pvalue;
}
void convCPU(float* h_a, float* h_mask, float*h_c_cpu, int mask_width, int width){
    
    for(int i=0; i<width; i++){
        
        float Pvalue = 0;
        int N_start_point= i - (mask_width / 2);

        for(int j=0; j<mask_width; j++){
            if ((N_start_point + j>=0) && (N_start_point+j < width)){
                Pvalue += h_a[N_start_point + j] * h_mask[j];
            }
        }
        h_c_cpu[i]=Pvalue;
    }

}

int main(){
    float *h_a, *h_mask, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_mask, *d_c;

    int n = 200, mask_len=120;

    int size_A = n * sizeof(float);
    int size_B = mask_len * sizeof(float);
    int size_C = n * sizeof(float);

    h_a = (float*)malloc(size_A);
    h_mask = (float*)malloc(size_B);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu = (float*)malloc(size_C);

    init_vector(h_a, n);
    init_vector(h_mask, mask_len);

    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_mask, size_B);
    cudaMalloc(&d_c, size_C);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size_B, cudaMemcpyHostToDevice);

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        convCPU(h_a, h_mask, h_c_cpu, mask_len, n);
        convGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_mask, d_c, mask_len, n);
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
        convCPU(h_a, h_mask, h_c_cpu, mask_len, n);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        convGPU<<<num_blocks, BLOCK_SIZE>>>(d_a, d_mask, d_c, mask_len, n);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);

    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    bool flag=true;

    for(int i=0; i<n; i++){
        if(std::abs(h_c_cpu[i]-h_c_gpu[i])>0.01){
            flag=false;
            break;
        }
    }

    printf("array equal: %d", flag);

    free(h_a);
    free(h_mask);
    free(h_c_cpu);
    free(h_c_gpu);

    cudaFree(d_a);
    cudaFree(d_mask);
    cudaFree(d_c);

    return 0;

}