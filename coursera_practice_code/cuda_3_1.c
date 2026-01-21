__global__ void convolution_1D_basic_kernel(float *N, float *M, float *P, 
                                            int Mask_Width, int Width) {

    // Calculate the global index for the current thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;
    // Calculate the starting point in the input array N for the current mask application
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++) {
        // Boundary condition handling: check if the index is within [0, Width)
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }

    // Write the result to the output array P
    P[i] = Pvalue;
}