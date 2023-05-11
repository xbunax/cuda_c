#include <stdio.h>
#include <stdlib.h>

// Define matrix dimensions
#define MATRIX_SIZE 1024

// CUDA kernel function to multiply matrices
__global__ void matrixMul(float *a, float *b, float *c, int matrix_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrix_size && col < matrix_size) {
        float sum = 0.0f;

        for (int i = 0; i < matrix_size; ++i) {
            sum += a[row * matrix_size + i] * b[i * matrix_size + col];
        }

        c[row * matrix_size + col] = sum;
    }
}

int main() {
    float *a, *b, *c; // Host matrices
    float *dev_a, *dev_b, *dev_c; // Device matrices

    // Allocate memory for host matrices
    a = (float *)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    b = (float *)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    c = (float *)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);

    // Read input matrices from files
    FILE *fileA = fopen("matrixA.txt", "r");
    FILE *fileB = fopen("matrixB.txt", "r");

    if (!fileA || !fileB) {
        printf("Error: Failed to open input files.\n");
        return 1;
    }
    else{
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        if (fscanf(fileA, "%f", &a[i]) != 1 || fscanf(fileB, "%f", &b[i]) != 1) {
            printf("Error: Failed to read input matrices.\n");
            return 1;
        }
    }
    }

    fclose(fileA);
    fclose(fileB);

    // Allocate memory for device matrices
    cudaMalloc((void **)&dev_a, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMalloc((void **)&dev_b, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMalloc((void **)&dev_c, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);

    // Copy host matrices to device matrices
    cudaMemcpy(dev_a, a, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(32, 32);
    dim3 gridDim((MATRIX_SIZE + blockDim.x - 1) / blockDim.x, (MATRIX_SIZE + blockDim.y - 1) / blockDim.y);

    // Launch kernel to multiply matrices
    matrixMul<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c, MATRIX_SIZE);

    // Copy result matrix from device to host
    cudaMemcpy(c, dev_c, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyDeviceToHost);

    // Write result matrix to file
    FILE *fileC = fopen("matrixC.txt", "w");

    if (!fileC) {
        printf("Error: Failed to open output file.\n");
        return 1;
    }
    else{
        for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            fprintf(fileC, "%f ", c[i * MATRIX_SIZE + j]);
        }
    }
    }
    fclose(fileC);
    

    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}