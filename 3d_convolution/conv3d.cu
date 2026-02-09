#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 8

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols, int channels, int num_matrices) {
    for(int matrix=0; matrix<num_matrices; matrix++){
        for (int i = 0; i < rows * cols * channels; i++) {
            mat[matrix*rows*cols*channels + i] = (float)rand() / RAND_MAX;
            // mat[i]=1.0f;
        }
    }
}

// CUDA kernel for 1 channel (3d conv)
__global__ 
void conv3dGPU(float* d_A, float* d_B, float* d_C, int height, int width, int channel, int mask_height , int mask_width, int mask_channel, int output_height, int output_width, int num_masks)
{

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int mask_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (Row<output_height && Col<output_width && mask_idx<num_masks){
        float output=0.0f;
        for(int i=0; i<mask_height; i++){
            for(int j=0; j<mask_width; j++){
                for(int k=0; k<mask_channel; k++){
                    output+=d_A[(Row+i)*width*channel + (Col+j)*channel + k] * d_B[mask_idx*mask_height*mask_width*mask_channel + i*mask_width*mask_channel + j*mask_channel + k];
                }
            }
        }
        d_C[mask_idx*output_height*output_width + Row*output_width + Col]=output;
    }

    __syncthreads();
}


void conv3dCPU(float* h_A, float* h_B, float* h_C, int height, int width, int channel, int mask_height , int mask_width, int mask_channel, int output_height, int output_width, int num_masks) 
{
    // chw
    for(int matrix=0; matrix<num_masks; matrix++){
        for(int row=0; row<output_height; row++){
            for(int col=0; col<output_width; col++){
                float sum=0.0f;
                for(int i=0; i<mask_height; i++){
                    for(int j=0; j<mask_width; j++){
                        for(int k=0; k<mask_channel; k++){
                            sum+=h_A[(row+i)*width*channel + (col+j)*channel + k]*h_B[matrix*mask_height*mask_width*mask_channel + i*mask_width*mask_channel + j*mask_channel + k];
                        }
                    }
                }
                h_C[matrix*output_height*output_width + row*output_width + col]=sum;
            }
        }
    }
}

// double get_time() {
//     struct timespec ts;
//     clock_gettime(CLOCK_MONOTONIC, &ts);
//     return ts.tv_sec + ts.tv_nsec * 1e-9;
// }

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    // Matrix sizes
    int height = 25, width = 25, channel = 8;
    int mask_width = 7, mask_height=7, mask_channel=8, num_masks = 8;

    int output_height = height - mask_height + 1;
    int output_width = width - mask_width + 1;
    int output_channel = num_masks;

    int size_A = height * width  * channel * sizeof(float);
    int size_B = num_masks * mask_height * mask_width * mask_channel * sizeof(float);
    int size_C = output_height * output_width  * output_channel * sizeof(float);

    h_a = (float*)malloc(size_A);
    h_b = (float*)malloc(size_B);
    h_c_cpu = (float*)malloc(size_C);
    h_c_gpu = (float*)malloc(size_C);

    // Host matrices (dummy data)
    init_matrix(h_a, height, width, channel, 1);
    init_matrix(h_b, mask_height, mask_width, mask_channel, num_masks);
    
    cudaMalloc(&d_a, size_A);
    cudaMalloc(&d_b, size_B);
    cudaMalloc(&d_c, size_C);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((output_width - 1) / BLOCK_SIZE + 1, (output_height - 1) / BLOCK_SIZE +1, (num_masks - 1) / BLOCK_SIZE + 1);

    printf("output height %d \n", output_height);
    printf("output width %d \n", output_width);
    printf("output channel %d \n", num_masks);

    // for(int i=0; i<height; i++){
    //     for(int j=0; j<width; j++){
    //         for(int k=0; k<channel; k++){
    //             printf("%f ", h_a[i*width*channel + j*channel + k]);
    //         }
    //     }
    //     printf("\n");
    // }

    // for(int i=0; i<mask_height; i++){
    //     for(int j=0; j<mask_width; j++){
    //         for(int k=0; k< mask_channel; k++){
    //             printf("%f ", h_b[i*mask_width*mask_channel + j*mask_channel + k]);
    //         }
    //     }
    //     printf("\n");
    // }

    // printf("Performing warm-up runs...\n");
    for (int i = 0; i < 1; i++) {
        conv3dCPU(h_a, h_b, h_c_cpu, height, width, channel, mask_height, mask_width, mask_channel, output_height, output_width, num_masks);
        conv3dGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, height, width, channel, mask_height, mask_width, mask_channel, output_height, output_width, num_masks);
        cudaDeviceSynchronize();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);

    // // Benchmark CPU implementation
    // printf("Benchmarking CPU implementation...\n");
    // double cpu_total_time = 0.0;
    // for (int i = 0; i < 20; i++) {
    //     double start_time = get_time();
    //     conv3dCPU(h_a, h_b, h_c_cpu, height, width, mask_width, output_height, output_width);
    //     double end_time = get_time();
    //     cpu_total_time += end_time - start_time;
    // }
    // double cpu_avg_time = cpu_total_time / 20.0;

    // for(int i=0; i<output_height; i++){
    //     for(int j=0; j<output_width; j++){
    //         printf("%f ", h_c_cpu[i*output_width+j]);
    //         // printf("%d ", 1);
    //     }
    //     printf("\n");
    // }

    // for(int i=0; i<output_height; i++){
    //     for(int j=0; j<output_width; j++){
    //         printf("%f ", h_c_gpu[i*output_width+j]);
    //         // printf("%d ", 1);
    //     }
    //     printf("\n");
    // }

    // // Benchmark GPU implementation
    // printf("Benchmarking GPU implementation...\n");
    // double gpu_total_time = 0.0;
    // for (int i = 0; i < 20; i++) {
    //     double start_time = get_time();
    //     conv3dGPU<<<gridDim, blockDim>>>(d_a, d_b, d_c, height, width, mask_width, output_height, output_width);
    //     cudaDeviceSynchronize();
    //     double end_time = get_time();
    //     gpu_total_time += end_time - start_time;
    // }
    // double gpu_avg_time = gpu_total_time / 20.0;

    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    // }

    // cudaDeviceSynchronize();

    // cudaMemcpy(h_c_gpu, d_c, size_C, cudaMemcpyDeviceToHost);

    // // Print results
    // printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    // printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    // printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    bool flag=true;

    for(int mask=0; mask<num_masks; mask++){
        for(int i=0; i<output_height; i++){
            for(int j=0; j<output_width; j++){
                if(std::abs(h_c_cpu[mask*output_height*output_width + i*output_width+j]-h_c_gpu[mask*output_height*output_width+i*output_width+j])>0.1){
                    flag=false;
                    break;
                }
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
