#ifndef MATRIX_INCLUDED
#define MATRIX_INCLUDED 1

__global__ void matrixMul(float *a, float *b, float *c, int matrix_size_row, int matrix_size_col);

__global__ void matrixSum(float * MatA,float * MatB,float * MatC,int nx,int ny);


#endif