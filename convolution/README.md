## CUDA Convolution Kernels
This directory contains multiple CUDA implementations of 1D, 2D, and 3D convolution, including optimized tiled versions and gradient/differential convolution variants.

#### Implemented Kernels

1. **1d_convolution/**

* Basic 1D convolution kernel.
* One thread computes one output element
* Direct global memory access
* Simple boundary checks


2. **1d_convolution_tiled/**

    Optimized 1D convolution using shared memory tiling.
* Reduced global memory reads
* Better memory coalescing
* Halo region loading

3. **2d_convolution_tiled/**

    Tiled 2D convolution (commonly used in image filtering).

* 2D grid and block configuration
* Shared memory tile loading
* Boundary handling

4. **3d_convolution/**
    Naive 3D convolution implementation.
* 3D grid/block indexing
* Volumetric data handling
* Direct global memory access

5. **3d_convolution_strided/**
* Strided output computation
* Reduced output size
* Efficient coordinate mapping

6. **diff_conv2d/**
* No padding
* No negative indexing
* Output smaller than input
* Simple memory access pattern

7. **diff_conv2d_stride/**
* Valid convolution with stride.
* Used for: Downsampling and Efficient feature extraction

8. **diff_conv2d_tiled/**
* Valid convolution optimized using shared memory.
* Improvements:Tile loading, Reduced global memory access, Higher performance

9. **diff_conv_tiled/**
* Advanced tiled valid convolution.
* Focus: Memory reuse, Efficient thread mapping, Reduced redundant reads

#### Compilation

Example:
````
nvcc -arch=sm_75 diff_conv2d/main.cu -o diffconv
./diffconv
````