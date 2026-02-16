
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#include "stb_image.h"
#include "stb_image_write.h"

#define BLOCK_SIZE 16

void RescaleImageCpu(unsigned char *imageInput, unsigned char *imageOutput, int width, int height, int output_height, int output_width)
{
    float yratio = ((float)height - 1) / ((float)output_height - 1);
    float xratio = ((float)width - 1) / ((float)output_width - 1);

    for (int y = 0; y < output_height; y++)
    {
        for (int x = 0; x < output_width; x++)
        {
            float y_in = yratio * y;
            float x_in = xratio * x;

            int y0 = floor(y_in);
            int y1 = min(y0 + 1, height - 1);

            int x0 = floor(x_in);
            int x1 = min(x0 + 1, width - 1);

            float dy = y_in - y0;
            float dx = x_in - x0;

            float n1 = (1 - dx) * (1 - dy) * imageInput[y0 * width + x0];
            float n2 = dx * (1 - dy) * imageInput[y0 * width + x1];
            float n3 = (1 - dx) * dy * imageInput[y1 * width + x0];
            float n4 = dx * dy * imageInput[y1 * width + x1];

            float V = n1 + n2 + n3 + n4;

            imageOutput[y * output_width + x] = V;
        }
    }
}

__global__ void RescaleImageGpu(unsigned char *imageInput, unsigned char *imageOutput, int width, int height, int output_height, int output_width)
{
    float yratio = ((float)height - 1) / ((float)output_height - 1);
    float xratio = ((float)width - 1) / ((float)output_width - 1);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height)
    {
        float y_in = yratio * y;
        float x_in = xratio * x;

        int y0 = floor(y_in);
        int y1 = min(y0 + 1, height - 1);

        int x0 = floor(x_in);
        int x1 = min(x0 + 1, width - 1);

        float dy = y_in - y0;
        float dx = x_in - x0;

        float n1 = (1 - dx) * (1 - dy) * imageInput[y0 * width + x0];
        float n2 = dx * (1 - dy) * imageInput[y0 * width + x1];
        float n3 = (1 - dx) * dy * imageInput[y1 * width + x0];
        float n4 = dx * dy * imageInput[y1 * width + x1];

        float V = n1 + n2 + n3 + n4;

        imageOutput[y * output_width + x] = V;
    }
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

    int output_height = 1024, output_width = 1024;

    std::cout << "Loading png file...";
    unsigned char *imageData = stbi_load(argv[1], &width, &height, &componentCount, 1);

    unsigned char *outputImageData = new unsigned char[output_width * output_height];
    unsigned char *ptrOutputImageDataGpu;

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

    // Process image on cpu
    // std::cout << "Processing image...";
    // RescaleImageCpu(imageData, outputImageData, width, height, output_width, output_height);
    // std::cout << " DONE" << std::endl;

    // Copy data to the gpu
    std::cout << "Copy data to GPU...";
    unsigned char *ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * sizeof(unsigned char)) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 1, cudaMemcpyHostToDevice) == cudaSuccess);
    // assert(cudaMalloc(&ptrImageDataGpu, output_width * output_height * 1) == cudaSuccess);
    assert(cudaMalloc(&ptrOutputImageDataGpu, output_width * output_height * 1) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Process image on gpu
    std::cout << "Running CUDA Kernel...";
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((output_width - 1) / BLOCK_SIZE + 1, (output_height - 1) / BLOCK_SIZE + 1);

    RescaleImageGpu<<<gridDim, blockDim>>>(ptrImageDataGpu, ptrOutputImageDataGpu, width, height, output_width, output_height);
    auto err = cudaGetLastError();
    std::cout << " DONE" << std::endl;

    // Copy data from the gpu
    std::cout << "Copy data from GPU...";
    assert(cudaMemcpy(outputImageData, ptrOutputImageDataGpu, output_width * output_height * 1, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Build output filename
    std::string fileNameOut = argv[1];

    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_rescaled_backward.png";

    // Write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png(fileNameOut.c_str(), output_width, output_height, 1, outputImageData, 1 * output_width);
    std::cout << " DONE";

    // Free memory
    // cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
    stbi_image_free(outputImageData);
}
