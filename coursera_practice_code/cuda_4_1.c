__shared__ float partialSum[2*BLOCK_SIZE];

unsigned int t = threadIdx.x;
unsigned int start = 2*blockIdx.x*blockDim.x;
partialSum[t] = input[start+t];
partialSum[blockDim+t]=input[start+blockDim.x+t];

for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) 
{
    __syncthreads();
    if (t % stride == 0) 
    {
        partialSum[2 * t] += partialSum[2 * t + stride];
    }
}