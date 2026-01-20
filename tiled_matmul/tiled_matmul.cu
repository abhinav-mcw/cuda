#include <cuda_runtime.h>
#include <iostream>

#define TILE_WIDTH 16

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

__global__ 
void tiledMatmulGPU(float* A, float* B, float* C, int m, int n, int k)
{
    // 1. & 2. Declare shared memory for the tiles of A and B
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    // 3. Identify the block indices
    int bx = blockIdx.x;  int by = blockIdx.y;
    
    // 4. Identify the thread indices within the block
    int tx = threadIdx.x; int ty = threadIdx.y;

    // 5. & 6. Calculate the global Row and Column indices for the element
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    // 7. Initialize the accumulator for the dot product
    float Cvalue = 0;
    
    // ... (The rest of the kernel logic would follow here)
    // 8. Loop over the A and B tiles required to compute the C element
    for (int t = 0; t < (n-1)/TILE_WIDTH+1; ++t) {

        // Collaborative loading of A and B tiles into shared memory
        // 9. Load an element from A into shared memory
        if (Row <m && t*TILE_WIDTH+tx<n ){
            ds_A[ty][tx] = A[Row*n + t*TILE_WIDTH + tx];
        } else{
            ds_A[ty][tx]=0.0;
        }
        
        // 10. Load an element from B into shared memory
        if(t*TILE_WIDTH+ty<n && Col<k){
            ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)*k + Col];
        } else{
            ds_B[ty][tx]=0.0;
        }

        // 11. Wait for all threads in the block to finish loading their elements
        __syncthreads();

        // 12. & 13. Compute the partial dot product for this tile
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }

        // 14. Synchronize again before starting the next tile iteration
        __syncthreads();
    }

    // 16. Write the final computed value to the global memory of matrix C
    if (Row<m && Col<k)
        C[Row*k + Col] = Cvalue;
}

int main()
{
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    // Matrix sizes
    int m = 32, n = 32, k = 32;

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


    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(
        (k - 1) / TILE_WIDTH +1,
        (m - 1) / TILE_WIDTH +1,
        1
    );

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmulCPU(h_a, h_b, h_c_cpu, m, n, k);
        tiledMatmulGPU<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
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
        tiledMatmulGPU<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);
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

    printf("flag: %d", flag);

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
