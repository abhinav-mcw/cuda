## CUDA Kernels – GPU Computation Examples
This repository contains hands-on CUDA C++ implementations of fundamental GPU computation patterns.

#### Implemented Kernels

1. **Vector Addition**

* Basic CUDA kernel
* Grid & block indexing
* Host ↔ Device memory transfer
* Performance comparison with CPU

2. **Matrix Multiplication**

* Row × Column multiplication
* One thread per output element
* Baseline implementation for optimization comparison

3. **Parallel Reduction**

* Sum reduction using GPU
* Tree-based reduction pattern
* Optimized memory access

4. **Convolution**

* Image filtering kernel
* Boundary handling
* Shared memory 

5. **Bilinear Image Rescaling**

* Forward bilinear interpolation
* GPU-based image upscaling
* Pixel mapping and interpolation weights

6. **Practice Implementations**

    Code inspired from:
* Coursera CUDA courses
* FreeCodeCamp CUDA tutorials

    Used for reinforcing:

* Kernel launch patterns
* Memory handling
* Debugging CUDA programs

#### Requirements

* NVIDIA GPU in colab
* CUDA Toolkit installed
* nvcc compiler

#### Compilation & Execution

Example:
````
nvcc -arch=sm_75 vector_addition.cu -o vector_add
./vector_add
````

#### References

* NVIDIA CUDA Programming Guide
* CUDA Best Practices Guide
* Official CUDA Samples
* Coursera CUDA Courses
* FreeCodeCamp GPU Programming Tutorials
* stdlib - https://github.com/nothings/stb