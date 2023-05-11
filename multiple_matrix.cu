#include <stdio.h>
#include <stdlib.h>

// Define matrix dimensions
#define MATRIX_SIZE_ROW 32
#define MATRIX_SIZE_COL 32

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


int main()
{
    float *a, *b, *c;             // Host matrices
    float *dev_a, *dev_b, *dev_c; // Device matrices

    a = (float *)malloc(sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW);
    b = (float *)malloc(sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW);
    c = (float *)malloc(sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW);

    // Allocate memory for host matrices
    FILE *filea = fopen("matrix_a.txt", "w");
    FILE *fileb = fopen("matrix_b.txt", "w");
    // Initialize host matrices with random values
    for (int i = 0; i < MATRIX_SIZE_ROW * MATRIX_SIZE_COL; ++i)
    {

        a[i] = (float)(rand() % 10);
        b[i] = (float)(rand() % 10);
        fprintf(filea, "%lf ", a[i]);
        fprintf(fileb, "%lf ", b[i]);
       if (i % MATRIX_SIZE_COL == 0 && i!=0)        {
            fprintf(filea, "\n");
            fprintf(fileb, "\n");
        }
    }
    fclose(filea);
    fclose(fileb);

    // Allocate memory for device matrices
    cudaMalloc((void **)&dev_a, sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW);
    cudaMalloc((void **)&dev_b, sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW);
    cudaMalloc((void **)&dev_c, sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW);
    // Copy host matrices to device matrices
    cudaMemcpy(dev_a, a, sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW, cudaMemcpyHostToDevice);
    ;
    // Define block and grid sizes
    dim3 blockDim(MATRIX_SIZE_COL, MATRIX_SIZE_ROW);
    dim3 gridDim((MATRIX_SIZE_COL + blockDim.x - 1) / blockDim.x, (MATRIX_SIZE_ROW + blockDim.y - 1) / blockDim.y);

    // Launch kernel to multiply matrices
    matrixMul<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c, MATRIX_SIZE_ROW, MATRIX_SIZE_COL);

    // Copy result matrix from device to host
    cudaMemcpy(c, dev_c, sizeof(float) * MATRIX_SIZE_COL * MATRIX_SIZE_ROW, cudaMemcpyDeviceToHost);

    FILE *filec = fopen("matrix_c", "w");

    for (int i = 0; i < MATRIX_SIZE_ROW*MATRIX_SIZE_COL; i++)
    {
        fprintf(filec, "%lf ", c[i]);
        // printf("c[%d]=%f",i,c[i]);
        if (i % MATRIX_SIZE_COL == 0 && i!=0)
        {
            fprintf(filec, "\n");
        }
    }
    fclose(filec);

    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
