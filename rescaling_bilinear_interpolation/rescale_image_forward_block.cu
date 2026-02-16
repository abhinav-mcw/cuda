
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <cmath>

#include "stb_image.h"
#include "stb_image_write.h"

// function to rescale image on cpu using bilinear interpolation foward mapping
void RescaleImageCpu(unsigned char *imageInput, unsigned char *imageOutput, int width, int height, int output_height, int output_width)
{
    // s is the scaling factor
    int s = output_height / height;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int us = s * x, vs = s * y;

            // du and dv are the offsets for the 2x2 neighborhood
            for (int du = 0; du < 2; du++)
            {
                for (int dv = 0; dv < 2; dv++)
                {
                    // Calculate the value for the output pixel using weighted input pixel value
                    float value = (1 - (float)du / 2) * (1 - (float)dv / 2) * imageInput[y * width + x];
                    imageOutput[(vs + dv) * output_width + (us + du)] += value;
                }
            }
        }
    }
}

// function to rescale image on gpu using bilinear interpolation foward mapping
__global__ void RescaleImageGpu(unsigned char *imageInput, unsigned char *imageOutput, int width, int height, int output_height, int output_width)
{
    // s is the scaling factor
    int s = output_height / height;

    // calculate the row and column index for the input image based on the block index
    int row = blockIdx.y;
    int col = blockIdx.x;

    int us = s * col, vs = s * row;

    // du and dv are the offsets for the 2x2 neighborhood
    int du = threadIdx.x;
    int dv = threadIdx.y;

    // Calculate the value for the output pixel using weighted input pixel value
    float value = (1 - (float)du / 2) * (1 - (float)dv / 2) * imageInput[row * width + col];
    imageOutput[(vs + dv) * output_width + (us + du)] += value;
}

int main(int argc, char **argv)
{
    // Check argument count
    if (argc < 2)
    {
        std::cout << "Usage: 03 Rescale <filename>";
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

    // // Process image on cpu
    // std::cout << "Processing image...";
    // RescaleImageCpu(imageData, outputImageData, width, height, output_width, output_height);
    // std::cout << " DONE" << std::endl;

    // Copy data to the gpu
    std::cout << "Copy data to GPU...";
    unsigned char *ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * sizeof(unsigned char)) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 1, cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMalloc(&ptrOutputImageDataGpu, output_width * output_height * 1) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Process image on gpu
    std::cout << "Running CUDA Kernel...";
    dim3 blockDim(2, 2);
    dim3 gridDim(256, 256);

    RescaleImageGpu<<<gridDim, blockDim>>>(ptrImageDataGpu, ptrOutputImageDataGpu, width, height, output_width, output_height);
    auto err = cudaGetLastError();
    std::cout << " DONE" << std::endl;

    // Copy data from the gpu
    std::cout << "Copy data from GPU...";
    assert(cudaMemcpy(outputImageData, ptrOutputImageDataGpu, output_width * output_height * 1, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Build output filename
    std::string fileNameOut = argv[1];

    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_forward_rescaled.png";

    // Write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png(fileNameOut.c_str(), output_width, output_height, 1, outputImageData, 1 * output_width);
    std::cout << " DONE";

    // Free memory
    // cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
    stbi_image_free(outputImageData);
}
