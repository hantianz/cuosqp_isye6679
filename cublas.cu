#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cublas_v2.h"
#include "support.h"


cublasHandle_t cublasHandle;

// return summation of two vectors
// C = A + B
void vecAdd(int n, double* d_A, double* d_B, double* d_C) {
    double alpha = 1;
    checkCublasErrors(cublasDcopy(cublasHandle, n, d_A, 1, d_C, 1));
    checkCublasErrors(cublasDaxpy(cublasHandle, n, &alpha, d_B, 1, d_C, 1));
}

// return inner project of two vectors
// C = A * B
double innerProduct(int n, double *d_A, double *d_B) {
    double h_C;
    checkCublasErrors(cublasDdot(cublasHandle, n, d_A, 1, d_B, 1, &h_C));
    return h_C;
}

// B = val * A
void scalarMulVec(int n, double val, double *d_A, double *d_B) {
    checkCudaErrors(cudaMemset(d_B, 0.0, sizeof(double)*n));
    checkCublasErrors(cublasDaxpy(cublasHandle, n, &val, d_A, 1, d_B, 1));
}

// scale a vector in place
// A = val * A
void scaleMulVecInPlace(int n, double val, double *d_A) {
    checkCublasErrors(cublasDscal(cublasHandle, n, &val, d_A, 1));
}


void vecAdd_cpu(int n, double *A, double *B, double *C){
    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}

double innerProduct_cpu(int n, double *A, double *B){
    double product = 0.0;
    for (int i = 0; i < n; ++i)
        product += A[i] * B[i];
    return product;
}

void scalarMulVec_cpu(int n, double val, double *A, double *B)
{
    for (int i = 0; i < n; ++i)
        B[i] = A[i] * val;
}

void verifyVecAdd(int n, double *A, double *B, double *C_gpu) {
  const float relativeTolerance = 1e-6;
  double C_cpu[n];
  vecAdd_cpu(n, A, B, C_cpu);

  for(int i = 0; i < n; i++) {
    double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
    if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
      printf("TEST FAILED\n");
      exit(0);
    }
  }
  printf("TEST PASSED\n");
}

void verifyInnerProduct(int n, double *A, double *B, double C_gpu) {
    const float relativeTolerance = 1e-6;
    double C_cpu = innerProduct_cpu(n, A, B);
    double relativeError = (C_cpu - C_gpu)/ C_cpu;
    if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
      printf("TEST FAILED\n");
      exit(0);
    }
  printf("TEST PASSED\n");
}

void verifyScaleMulVec(int n, double val, double *A, double* C_gpu) {
  const float relativeTolerance = 1e-6;
  double C_cpu[n];
  scalarMulVec_cpu(n, val, A, C_cpu);

  for(int i = 0; i < n; i++) {
    double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
    if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
      printf("TEST FAILED\n");
      exit(0);
    }
  }
  printf("TEST PASSED\n");
}


int main(int argc, char**argv) {
    checkCublasErrors(cublasCreate(&cublasHandle));

    unsigned int n = 10000;

    double* h_A = (double*) malloc( sizeof(double)*n );
    for (unsigned int i=0; i < n; i++) { h_A[i] = (rand()%100)/100.00;}

    double* h_B = (double*) malloc( sizeof(double)*n );
    for (unsigned int i=0; i < n; i++) { h_B[i] = (rand()%100)/100.00;}

    double* h_C = (double*) malloc( sizeof(double)*n );
    
    double* d_A;
    checkCudaErrors(cudaMalloc((void**) &d_A, sizeof(double)*n));

    double* d_B;
    checkCudaErrors(cudaMalloc((void**) &d_B, sizeof(double)*n));

    double* d_C;
    checkCudaErrors(cudaMalloc((void**) &d_C, sizeof(double)*n));
    
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(double)*n, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, sizeof(double)*n, cudaMemcpyHostToDevice));


    printf("- Test vecAdd\n");
    vecAdd(n, d_A, d_B, d_C);
    checkCudaErrors(cudaMemcpy(h_C, d_C, sizeof(double)*n, cudaMemcpyDeviceToHost));
    verifyVecAdd(n, h_A, h_B, h_C);

    printf("- Test innerProduct\n");
    double h_C_scalar = 0;
    h_C_scalar = innerProduct(n, d_A, d_B);
    verifyInnerProduct(n, h_A, h_B, h_C_scalar);

    printf("- Test scalarMulVec\n");
    double val = 3.1415;
    scalarMulVec(n, val, d_A, d_C);
    checkCudaErrors(cudaMemcpy(h_C, d_C, sizeof(double)*n, cudaMemcpyDeviceToHost));
    verifyScaleMulVec(n, val, h_A, h_C);

    free(h_A);
    free(h_B);
    free(h_C);

    //INSERT CODE HERE to free device matrices
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(cublasHandle);
    return 0;

}


// /* CSR matrix format */
// struct csr {
//     int m;          // number of rows
//     int n;          // number of columns
//     int nnz; // the number of nonzero elements in the matrix
//     double *csrVal; // points to the data array of length nnz that holds all nonzero values of A in row-major format.
//     int *csrRowPtr; // points to the integer array of length m+1 that holds indices into the arrays csrColIndA and csrValA.
//     int *csrColInd; // points to the integer array of length nnz that contains the column indices of the corresponding elements in array csrValA.
//     cusparseMatDescr_t  matDescr;
// };


// // calculate matrix A* vector B and store it in vector C
// // A [m x n] is a sparse matrix, B [n x 1] is a dense vector, C [m x 1] is a dense vector,
// // C = A * B
// void matMulVec(int n, int m, csr h_A, double *h_B, double *h_C) {
//     double alpha = 1;
//     double beta = 1;
    
//     //prepare buffer
//     char *buffer;
//     size_t BufferSizeInBytes;

//     cusparseSpMV_bufferSize(cusparseHandle, 
//                  CUSPARSE_OPERATION_NON_TRANSPOSE, 
//                  &alpha,
//                  h_A->matDescr,
//                  h_B->vecDescr,
//                  &beta,
//                  h_C->vecDescr,
//                  CUDA_R_64F,
//                  CUSPARSE_SPMV_ALG_DEFAULT,
//                  BufferSizeInBytes,
//                 )

//     cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes);

//     cusparseSpMV(cusparseHandle, 
//                  CUSPARSE_OPERATION_NON_TRANSPOSE, 
//                  &alpha,
//                  h_A->matDescr,
//                  h_B->vecDescr,
//                  &beta,
//                  h_C->vecDescr,
//                  CUDA_R_64F,
//                  CUSPARSE_SPMV_ALG_DEFAULT,
//                  buffer
//                 );
//     cudaFree(buffer);
// }

// // calculate A*B and store it in C
// // A, B C are sparse matrices
// void matMulMat(csr *A, csr *B, csr *C) {


// }


// calculate sum of two matrices
// A [m x n], B[m, n], C[m, n] are sparse matrices
// C = A + B

// void matAdd(int n, int m, double A[n][m], double B[n][m], int C[n][m]) {
//     int baseC, nnzC;
//     /* alpha, nnzTotalDevHostPtr points to host memory */
//     size_t BufferSizeInBytes;
//     char *buffer = NULL;
//     int *nnzTotalDevHostPtr = &nnzC;
//     cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
//     cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
//     /* prepare buffer */
//     cusparseScsrgeam2_bufferSizeExt(handle, m, n,
//         alpha,
//         descrA, nnzA,
//         csrValA, csrRowPtrA, csrColIndA,
//         beta,
//         descrB, nnzB,
//         csrValB, csrRowPtrB, csrColIndB,
//         descrC,
//         csrValC, csrRowPtrC, csrColIndC
//         &bufferSizeInBytes
//         );
//     cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes);
//     cusparseXcsrgeam2Nnz(handle, m, n,
//             descrA, nnzA, csrRowPtrA, csrColIndA,
//             descrB, nnzB, csrRowPtrB, csrColIndB,
//             descrC, csrRowPtrC, nnzTotalDevHostPtr,
//             buffer);
//     if (NULL != nnzTotalDevHostPtr){
//         nnzC = *nnzTotalDevHostPtr;
//     }else{
//         cudaMemcpy(&nnzC, csrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
//         cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
//         nnzC -= baseC;
//     }
//     cudaMalloc((void**)&csrColIndC, sizeof(int)*nnzC);
//     cudaMalloc((void**)&csrValC, sizeof(double)*nnzC);
//     cusparseScsrgeam2(handle, m, n,
//             alpha,
//             descrA, nnzA,
//             csrValA, csrRowPtrA, csrColIndA,
//             beta,
//             descrB, nnzB,
//             csrValB, csrRowPtrB, csrColIndB,
//             descrC,
//             csrValC, csrRowPtrC, csrColIndC
//             buffer);
// }
