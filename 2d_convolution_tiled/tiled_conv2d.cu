#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH+4)

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
        // mat[i]=1.0f;
    }
}

// CUDA kernel for 1 channel (2d conv)
__global__ 
void conv2dGPU(float* d_A, float* d_B, float* d_C, int height, int width, int mask_width)
{
    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;

    int row_i = row_o - 2;
    int col_i = col_o - 2;

    if ((row_i>=0) && (row_i < height) && (col_i>=0) && (col_i<width)){
        Ns[ty][tx] = d_A[row_i*width + col_i];
    } else {
        Ns[ty][tx] = 0.0f;
    }

    __syncthreads();

    float output = 0.0f;

    if (ty<O_TILE_WIDTH && tx<O_TILE_WIDTH){
        for(int i=0; i<mask_width; i++){
            for(int j=0; j<mask_width; j++){
                output+=d_B[i*mask_width+j] * Ns[i+ty][j+tx];
            }
        }
    }

    if(row_o<height && col_o < width)
        d_C[row_o*width + col_o]=output;

    __syncthreads();
}


void conv2dCPU(float* h_A, float* h_B, float* h_C, int height, int width, int mask_width) 
{
    for(int row=0; row<height; row++){
        for(int col=0; col<width; col++){
            float sum=0.0f;
            int start_row = row - mask_width/2;
            int start_col = col - mask_width/2;
            for(int i=0; i<mask_width; i++){
                for(int j=0; j<mask_width; j++){
                    if ( (start_row+i>=0) && (start_row+i < height) && (start_col+j >=0 ) && (start_col +j <width)) 
                        sum+=h_B[i*mask_width + j]* h_A[(start_row+i)*width + (start_col+j)];
                }
            }
            h_C[row*width+col]=sum;
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
    int height = 128, width = 128, channel = 1;
    int mask_width = 5;

    int size_A = height * width * channel * sizeof(float);
    int size_B = mask_width * mask_width * sizeof(float);
    int size_C = height * width * channel * sizeof(float);

    h_a = (float*)malloc(size_A);
    h_b = (float*)malloc(size_B);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu = (float*)malloc(size_C);

    // Host matrices (dummy data)
    init_matrix(h_a, height, width); //channel = 1
    init_matrix(h_b, mask_width, mask_width);

    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);


    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim((width - 1) / O_TILE_WIDTH + 1, (height - 1) / O_TILE_WIDTH +1);

    // conv2dCPU(h_a, h_b, h_c_cpu, height, width, mask_width);

    // for(int i=0; i<height; i++){
    //     for(int j=0; j<width; j++){
    //         printf("%f ",h_c_cpu[i*width+j]);
    //     }
    //     printf("\n");
    // }

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        conv2dCPU(h_a, h_b, h_c_cpu, height, width, mask_width);
        conv2dGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, height, width, mask_width);
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
        conv2dCPU(h_a, h_b, h_c_cpu, height, width, mask_width);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        conv2dGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, height, width, mask_width);
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

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            if(std::abs(h_c_cpu[i*width+j]-h_c_gpu[i*width+j])>0.0001){
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
