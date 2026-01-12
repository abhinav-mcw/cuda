int dev_count;
cudaGetDeviceCount( &dev_count );

cudaDeviceProp dev_prop;

for (int i = 0; i < dev_count; i++) {
    cudaGetDeviceProperties( &dev_prop, i );

    // decide if device has sufficient resources and capabilities
}