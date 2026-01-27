#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define O_TILE_WIDTH 60
#define BLOCK_WIDTH (O_TILE_WIDTH + 4)

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
void convGPUTiled(float *N, float *M, float *P, int Mask_Width, int Width) {

    int tx = threadIdx.x;

    // Calculate the global index for the current thread
    int index_o=blockIdx.x*O_TILE_WIDTH + tx;

    int n=Mask_Width/2;
    int index_i=index_o-n;

    __shared__ float Ns[BLOCK_WIDTH];

    if ((index_i>=0)&& (index_i < Width)){
        Ns[tx]=N[index_i];
    }
    else{
        Ns[tx]=0.0f;
    }

    __syncthreads();

    if (threadIdx.x < O_TILE_WIDTH) {
        float output = 0.0f;
        for(int j = 0; j < Mask_Width; j++) {
            output += M[j] * Ns[j + tx];
        }
        P[index_o] = output;
    }

    __syncthreads();


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

    int n = 1024, mask_len=5;

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

    dim3 dimBlock(BLOCK_WIDTH, 1, 1);

    dim3 dimGrid((n-1)/O_TILE_WIDTH+1, 1, 1);


    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        convCPU(h_a, h_mask, h_c_cpu, mask_len, n);
        convGPUTiled<<<dimGrid, dimBlock>>>(d_a, d_mask, d_c, mask_len, n);
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
        convGPUTiled<<<dimGrid, dimBlock>>>(d_a, d_mask, d_c, mask_len, n);
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