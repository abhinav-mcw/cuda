int index_o=blockIdx.x*O_TILE_WIDTH + threadIdx.x;

int n=Mask_Width/2;
int index_i=index_o-n;

float output=0.0f;

if ((index_i>=0)&& (index_i < Width)){
    Ns[tx]=N[index_i];
}
else{
    Ns[tx]=0.0f;
}


if (threadIdx.x < O_TILE_WIDTH) {
    output = 0.0f;
    for(j = 0; j < Mask_Width; j++) {
        output += M[j] * Ns[j + threadIdx.x];
    }
    P[index_o] = output;
}

#define O_TILE_WIDTH 1020
#define BLOCK_WIDTH (O_TILE_WIDTH + 4)

dim3 dimBlock(BLOCK_WIDTH, 1, 1);

dim3 dimGrid((Width-1)/O_TILE_WIDTH+1, 1, 1)

/* The Mask_Width is 5 in this example
   In general, block width should be 
       output tile width + (mask width-1)
*/