// hw intellec property - video decoding, sound processing
// dsp core			- computing


// cpu - less registers for less threads
// gpu - more registers for more threads


void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n * sizeof(float); 
    float* d_A, *d_B, *d_C;

    cudaMalloc((void **) &d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) &d_C, size);

    // Kernel invocation code - to be shown later

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

cudaError_t  err = cudaMalloc((void **) &d_A, size);

if (err != cudaSuccess) {
    printf("%s in %s at line %d\n",
           cudaGetErrorString(err), __FILE__,
           __LINE__);
    exit(EXIT_FAILURE);
}


