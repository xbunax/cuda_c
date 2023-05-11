#include <stdio.h>
#include <stdlib.h>



// CUDA kernel function to multiply matrices
__global__ void matrixMul(float *a, float *b, float *c, int matrix_size_row, int matrix_size_col)
{

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < matrix_size_row && col < matrix_size_col) {
        float sum = 0.0f;

        for (int i = 0; i < matrix_size_col; ++i) {
            sum += a[row * matrix_size_col + i] * b[i * matrix_size_col+ col];
        }

        c[row * matrix_size_col + col] = sum;
    }
}
