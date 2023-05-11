#include <stdio.h>
#include <stdlib.h>

// __device__ float getElement(Matrix *A, int row, int col);
// __device__ void setElement(Matrix *A, int row, int col, float value);

struct Matrix
{
    int width;
    int height;
    float *elements;
};
__device__ float getElement(Matrix *A, int row, int col)
{
    return A->elements[row * A->width + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix *A, int row, int col, float value)
{
    A->elements[row * A->width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素
__global__ void matMulKernel(Matrix *A, Matrix *B, Matrix *C)
{
    float Cvalue = 0.0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < A->width; ++i)
    {
        Cvalue += getElement(A, row, i) * getElement(B, i, col);
    }
    setElement(C, row, col, Cvalue);
}

int main()
{
    //     struct Matrix
    // {
    //     int width;
    //     int height;
    //     float *elements;
    // };

    int width = 1 << 10;
    int height = 1 << 10;
    Matrix *A, *B, *C;
    // 申请托管内存
    cudaMallocManaged((void **)&A, sizeof(Matrix));
    cudaMallocManaged((void **)&B, sizeof(Matrix));
    cudaMallocManaged((void **)&C, sizeof(Matrix));
    int nBytes = width * height * sizeof(float);
    cudaMallocManaged((void **)&A->elements, nBytes);
    cudaMallocManaged((void **)&B->elements, nBytes);
    cudaMallocManaged((void **)&C->elements, nBytes);

    // 初始化数据
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for (int i = 0; i < width * height; ++i)
    {
        A->elements[i] = (float)(rand() % 10);
        B->elements[i] = (float)(rand() % 10);
    }

    // 定义kernel的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    // 执行kernel
    matMulKernel<<<gridSize, blockSize>>>(A, B, C);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();

    // for (int i = 0; i < height*width; i++)
    // {
    //     {
    //         printf("c[%d]=%f", i, C->elements[i]);
    //     }
    // }

    FILE *filec = fopen("matrix_c_test", "w");

    for (int i = 0; i < width*height; i++)
    {
        fprintf(filec, "%lf ", C->elements[i]);
        // printf("c[%d]=%f",i,c[i]);
        if (i % width == 0 && i!=0)
        {
            fprintf(filec, "\n");
        }
    }

    // 检查执行结果
    // float maxError = 0.0;
    // for (int i = 0; i < width * height; ++i)
    //     maxError = fmax(maxError, fabs(C->elements[i] - 2 * width));
    // std::cout << "最大误差: " << maxError << std::endl;

    return 0;
}