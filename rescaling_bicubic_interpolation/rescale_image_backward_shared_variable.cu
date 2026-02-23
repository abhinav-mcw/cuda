
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#include "stb_image.h"
#include "stb_image_write.h"

#define BLOCK_SIZE 16

using namespace std;

int clamp(int i, int m, int limit)
{
    // clamp i+m between 0 and limit-1
    int pos = 0;
    if (i + m < 0)
        pos = 0;
    else if (i + m >= limit)
        pos = limit - 1;
    else
        pos = i + m;
    return pos;
}

void assignWeights(float *weightsArray, float u)
{
    float t = 0;
    float a = -0.5;

    // loop to store weights for m - u
    for (int m = -1; m < 3; m++)
    {
        t = m - u;
        if (abs(t) < 1)
        {
            weightsArray[m + 1] = (a + 2) * pow(abs(t), 3) - (a + 3) * pow(abs(t), 2) + 1;
        }
        else if (abs(t) >= 1 && abs(t) < 2)
        {
            weightsArray[m + 1] = a * pow(abs(t), 3) - 5 * a * pow(abs(t), 2) + 8 * a * abs(t) - 4 * a;
        }
        else if (abs(t) >= 2)
        {
            weightsArray[m + 1] = 0;
        }
    }
}

// function to rescale image on cpu using bicubic interpolation backward mapping
void RescaleImageCpu(unsigned char *imageInput, unsigned char *imageOutput, int width, int height, int output_height, int output_width)
{
    // calculate scale factor for x and y
    float Sx = (float)output_width / width;
    float Sy = (float)output_height / height;

    // declare weights array to store weights
    float weights_x[4];
    float weights_y[4];

    // loop for output image matrix pixel calculations
    for (int row = 0; row < output_height; row++)
    {
        for (int col = 0; col < output_width; col++)
        {
            // calculate input pixel point
            float xs = col / Sx;
            float ys = row / Sy;

            // calculate points
            int i = (int)floor(xs);
            int j = (int)floor(ys);

            // find delta
            float u = xs - i;
            float v = ys - j;

            // store weights for m and n in 2 arrays
            assignWeights(weights_x, u);
            assignWeights(weights_y, v);

            float weighted_sum = 0;

            // calculate weighted sum of 16 neighbouring pixels
            for (int n = -1; n < 3; n++)
            {
                for (int m = -1; m < 3; m++)
                {
                    weighted_sum += imageInput[clamp(j, n, height) * width + clamp(i, m, width)] * weights_x[m + 1] * weights_y[n + 1];
                }
            }

            if (weighted_sum < 0)
                weighted_sum = 0;
            else if (weighted_sum > 255)
                weighted_sum = 255;

            // store the weighted sum for the output pixel
            imageOutput[row * output_width + col] = weighted_sum;
        }
    }
}

__device__ int clampGpu(int i, int m, int limit)
{
    // clamp i+m between 0 and limit-1
    int pos = 0;
    if (i + m < 0)
        pos = 0;
    else if (i + m >= limit)
        pos = limit - 1;
    else
        pos = i + m;
    return pos;
}

__device__ void assignWeightsGpu(float *weightsArray, float u)
{
    float t = 0;
    float a = -0.5;

    // loop to store weights for m-u
    for (int m = -1; m < 3; m++)
    {
        t = m - u;
        if (abs(t) < 1)
        {
            weightsArray[m + 1] = (a + 2) * pow(abs(t), 3) - (a + 3) * pow(abs(t), 2) + 1;
        }
        else if (abs(t) >= 1 && abs(t) < 2)
        {
            weightsArray[m + 1] = a * pow(abs(t), 3) - 5 * a * pow(abs(t), 2) + 8 * a * abs(t) - 4 * a;
        }
        else if (abs(t) >= 2)
        {
            weightsArray[m + 1] = 0;
        }
    }
}

// function to rescale image on gpu using bilinear interpolation backward mapping
__global__ void RescaleImageGpu(unsigned char *imageInput, unsigned char *imageOutput, int width, int height, int output_height, int output_width)
{
    // calculate scale factor for x and y
    float Sx = (float)output_width / width;
    float Sy = (float)output_height / height;

    // declare weights array to store weights
    float weights_x[4];
    float weights_y[4];

    // global row and col index for output pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < output_height && col < output_width)
    {
        // calculate input pixel point
        float xs = col / Sx;
        float ys = row / Sy;

        // calculate points
        int i = (int)floor(xs);
        int j = (int)floor(ys);

        // find delta
        float u = xs - i;
        float v = ys - j;

        // store weights for m and n in 2 arrays
        assignWeightsGpu(weights_x, u);
        assignWeightsGpu(weights_y, v);

        float weighted_sum = 0;

        // calculate weighted sum of 16 neighbouring pixels
        for (int n = -1; n < 3; n++)
        {
            for (int m = -1; m < 3; m++)
            {
                weighted_sum += imageInput[clampGpu(j, n, height) * width + clampGpu(i, m, width)] * weights_x[m + 1] * weights_y[n + 1];
            }
        }

        if (weighted_sum < 0)
            weighted_sum = 0;
        else if (weighted_sum > 255)
            weighted_sum = 255;

        // store the weighted sum for the output pixel
        imageOutput[row * output_width + col] = weighted_sum;
    }
}

// function to rescale image on gpu using bilinear interpolation backward mapping
__global__ void RescaleImageGpuBlock(unsigned char *imageInput, unsigned char *imageOutput, int width, int height, int output_height, int output_width)
{
    // calculate scale factor for x and y
    float Sx = (float)output_width / width;
    float Sy = (float)output_height / height;

    // declare weights array to store weights
    float weights_x[4];
    float weights_y[4];

    // global row and col index for output pixel
    int col = blockIdx.x;
    int row = blockIdx.y;

    if (row < output_height && col < output_width)
    {
        // calculate input pixel point
        float xs = col / Sx;
        float ys = row / Sy;

        // calculate points
        int i = (int)floor(xs);
        int j = (int)floor(ys);

        // find delta
        float u = xs - i;
        float v = ys - j;

        // store weights for m and n in 2 arrays
        assignWeightsGpu(weights_x, u);
        assignWeightsGpu(weights_y, v);

        // declaring variable for shared memory
        __shared__ float weighted_sum;

        // 16 thread indices for computation
        int n = threadIdx.y;
        int m = threadIdx.x;

        // first thread initializes sum to 0
        if (n == 0 && m == 0)
        {
            weighted_sum = 0.0f;
        }
        __syncthreads();

        // computed weighted sum
        float value = imageInput[clampGpu(j, n - 1, height) * width + clampGpu(i, m - 1, width)] * weights_x[m] * weights_y[n];

        // weighted_sum += value;
        atomicAdd(&weighted_sum, value);

        __syncthreads();

        // 1 thread writes the output of the weighted sum for output pixel
        if (n == 0 && m == 0)
        {
            if (weighted_sum < 0)
                weighted_sum = 0;
            else if (weighted_sum > 255)
                weighted_sum = 255;
            imageOutput[row * output_width + col] = weighted_sum;
        }
    }
}

double get_time()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
    // Check argument count
    if (argc < 2)
    {
        std::cout << "Usage: 03_Rescale <filename>";
        return -1;
    }

    // Open image
    int width, height, componentCount;
    width = 256;
    height = 256;

    int output_height = 2048, output_width = 2048;

    std::cout << "Loading png file...";
    unsigned char *imageData = stbi_load(argv[1], &width, &height, &componentCount, 1);

    unsigned char *outputImageData = new unsigned char[output_width * output_height];
    unsigned char *ptrOutputImageDataGpu;
    unsigned char *ptrOutputImageDataGpuBlock;

    if (!imageData)
    {
        std::cout << std::endl
                  << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        std::cout << "Width and/or Height is not dividable by 32!";
        return -1;
    }

    // // Process image on cpu
    // std::cout << "Processing image...";
    // RescaleImageCpu(imageData, outputImageData, width, height, output_width, output_height);
    // std::cout << " DONE" << std::endl;

    // Copy data to the gpu
    std::cout << "Copy data to GPU...";
    unsigned char *ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * sizeof(unsigned char)) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 1, cudaMemcpyHostToDevice) == cudaSuccess);

    // assert(cudaMalloc(&ptrOutputImageDataGpu, output_width * output_height) == cudaSuccess);

    std::cout << " DONE" << std::endl;

    // Process image on gpu
    std::cout << "Running CUDA Kernel...";
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((output_width - 1) / BLOCK_SIZE + 1, (output_height - 1) / BLOCK_SIZE + 1);

    // RescaleImageGpu<<<gridDim, blockDim>>>(ptrImageDataGpu, ptrOutputImageDataGpu, width, height, output_width, output_height);
    // auto err = cudaGetLastError();
    // std::cout << " DONE" << std::endl;

    assert(cudaMalloc(&ptrOutputImageDataGpuBlock, output_width * output_height) == cudaSuccess);

    // Process image on gpu
    std::cout << "Running CUDA Kernel...";
    dim3 blockDim1(4, 4);
    dim3 gridDim1(output_height, output_width);

    RescaleImageGpuBlock<<<gridDim1, blockDim1>>>(ptrImageDataGpu, ptrOutputImageDataGpuBlock, width, height, output_width, output_height);
    auto err = cudaGetLastError();
    std::cout << " DONE" << std::endl;

    // Copy data from the gpu
    std::cout << "Copy data from GPU...";
    assert(cudaMemcpy(outputImageData, ptrOutputImageDataGpuBlock, output_width * output_height, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Build output filename
    std::string fileNameOut = argv[1];

    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_rescaled_backward_gpu_block.png";

    // Write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png(fileNameOut.c_str(), output_width, output_height, 1, outputImageData, 1 * output_width);
    std::cout << " DONE" << endl;

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 10; i++)
    {
        RescaleImageGpu<<<gridDim, blockDim>>>(ptrImageDataGpu, ptrOutputImageDataGpu, width, height, output_width, output_height);
        cudaDeviceSynchronize();
        RescaleImageGpuBlock<<<gridDim1, blockDim1>>>(ptrImageDataGpu, ptrOutputImageDataGpuBlock, width, height, output_width, output_height);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        RescaleImageGpu<<<gridDim, blockDim>>>(ptrImageDataGpu, ptrOutputImageDataGpu, width, height, output_width, output_height);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    printf("Benchmarking GPU block implementation...\n");
    double gpu_block_total_time = 0.0;
    for (int i = 0; i < 20; i++)
    {
        double start_time = get_time();
        RescaleImageGpuBlock<<<gridDim1, blockDim1>>>(ptrImageDataGpu, ptrOutputImageDataGpuBlock, width, height, output_width, output_height);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_block_total_time += end_time - start_time;
    }
    double gpu_block_avg_time = gpu_block_total_time / 20.0;

    printf("GPU average time: %f milliseconds\n", gpu_avg_time * 1000);
    printf("GPU block average time: %f milliseconds\n", gpu_block_avg_time * 1000);

    // Free memory
    // cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
    stbi_image_free(outputImageData);
}
