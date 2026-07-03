#include <cuda_runtime.h>
#include <stdio.h>

__device__ void printMatrix(const float* input, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f", input[rows * j + i]);
        }
    }
}

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols)
{
    const int WIDTH = 16;
    __shared__ float in[WIDTH][WIDTH];
    //__shared__ float out[WIDTH][WIDTH];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows)
    {
        in[threadIdx.y][threadIdx.x] = input[cols * row + col];
    }
    __syncthreads();
    if (col < cols && row < rows)
        //output[rows * col + row] = in[threadIdx.y][threadIdx.x];
        output[rows * col + row] = input[cols * row + col];
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
