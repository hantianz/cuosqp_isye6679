// nvcc cusparse.cu  -lcusparse  -o  cusparse
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "support.h"
#include "cublas_v2.h"
#include <cusparse.h>

void checkCudaErrors(cudaError_t cuda_ret) {
    if(cuda_ret != cudaSuccess) FATAL("CUDA Error");
}
void checkCublasErrors(cublasStatus_t cuda_ret) {
    if(cuda_ret != CUBLAS_STATUS_SUCCESS) FATAL("Cublas Error");
}

void checkCusparseErrors(cusparseStatus_t cuda_ret) {
    if(cuda_ret != CUSPARSE_STATUS_SUCCESS) {
        FATAL("Cusparse Error: %s", cusparseGetErrorString(cuda_ret));
    }
}

cusparseHandle_t cusparseHandle;
cublasHandle_t cublasHandle;


/* ------------------------------- type define--------------------------------*/

/* CSR matrix */
typedef struct {
    int m; // number of rows
    int n; //num of columns
    int nnz; // num of nonzero entries
    double* h_val; // Points to the data array of length nnz that holds all nonzero values of A in row-major format.
    int* h_rowPtr; // Points to the integer array of length m+1 that holds indices into the arrays csrColIndA and csrValA
    int* h_colInd; // Points to the integer array of length nnz that contains the column indices of the corresponding elements in array csrValA
} CSR_h;

typedef struct {
    int m; // number of rows
    int n; //num of columns
    int nnz; // num of nonzero entries
    double* d_val;
    int* d_rowPtr;
    int* d_colInd; 
    cusparseSpMatDescr_t descr;
} CSR_d;

void initCSR_h(CSR_h *mat, int m, int n, int nnz, double *h_val, int *h_rowPtr, int *h_colInd) {
    mat->m = m;
    mat->n = n;
    mat->nnz = nnz;
    mat->h_val = h_val;
    mat->h_rowPtr = h_rowPtr;
    mat->h_colInd = h_colInd;
}

void initCSR_d(CSR_d *mat, int m, int n, int nnz, double *d_val, int *d_rowPtr, int *d_colInd) {
    mat->m = m;
    mat->n = n;
    mat->nnz = nnz;

    mat->d_val = d_val;
    mat->d_rowPtr = d_rowPtr;
    mat->d_colInd = d_colInd;

    if(nnz > 0) {
        checkCusparseErrors(cusparseCreateCsr(&mat->descr, 
                          m, 
                          n,
                          nnz,
                          d_rowPtr,
                          d_colInd,
                          d_val,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, 
                          CUDA_R_64F));
    }

}

/* Dense vector*/
typedef struct {
    int n; // length of vector
    double* h_val;
} VEC_h;


typedef struct {
    int n; // length of vector
    double* d_val;
    cusparseDnVecDescr_t descr;
} VEC_d;

void initVEC_h(VEC_h *v, int n, double *h_val) {
    v->n = n;
    v->h_val = h_val;
}

void initVEC_d(VEC_d *v, int n, double *d_val) {
    v->n = n;
    v->d_val = d_val;
    checkCusparseErrors(cusparseCreateDnVec(&v->descr, n, d_val, CUDA_R_64F));
}


/* Dense matrix*/
typedef struct {
    int m;
    int n;
    double* h_val;
} DN_h;

typedef struct {
    int m;
    int n;
    double* d_val;
    cusparseDnMatDescr_t descr;
} DN_d;

void initDN_h(DN_h *mat, int m, int n, double *h_val) {
    mat->m = m;
    mat->n = n;
    mat->h_val = h_val;
}

void initDN_d(DN_d *mat, int m, int n, double *d_val) {
    mat->m = m;
    mat->n = n;
    mat->d_val = d_val;
    checkCusparseErrors(cusparseCreateDnMat(&mat->descr, m, n, m, mat->d_val, CUDA_R_64F, CUSPARSE_ORDER_ROW));
}


/* ------------------------------- type conversion -------------------------------*/

void CSR_h2d(CSR_h *h_mat, CSR_d *d_mat) {
    double *d_val;
    int *d_rowPtr;
    int *d_colInd;

    if(h_mat->nnz > 0) {
        checkCudaErrors(cudaMalloc((void**)&d_val, sizeof(double)*h_mat->nnz));
        checkCudaErrors(cudaMalloc((void**)&d_rowPtr, sizeof(int)*(h_mat->m+1)));
        checkCudaErrors(cudaMalloc((void**)&d_colInd, sizeof(int)*h_mat->nnz));

        checkCudaErrors(cudaMemcpy(d_val, h_mat->h_val, sizeof(double)*h_mat->nnz, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_rowPtr, h_mat->h_rowPtr, sizeof(int)*(h_mat->m+1), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_colInd, h_mat->h_colInd, sizeof(int)*h_mat->nnz, cudaMemcpyHostToDevice));
    }
    else{
        d_val = NULL;
        d_colInd = NULL;
        d_rowPtr = NULL;
        // checkCudaErrors(cudaMalloc((void**)&d_rowPtr, sizeof(int)*(h_mat->m+1)));
        // checkCudaErrors(cudaMemcpy(d_rowPtr, h_mat->h_rowPtr, sizeof(int)*(h_mat->m+1), cudaMemcpyHostToDevice));
    }

    initCSR_d(d_mat, h_mat->m, h_mat->n, h_mat->nnz, d_val, d_rowPtr, d_colInd);
    
}

void CSR_d2h(CSR_d *d_mat, CSR_h *h_mat) {
    double *h_val = (double *) malloc(sizeof(double)*d_mat->nnz);
    int *h_rowPtr = (int *) malloc(sizeof(int)*(d_mat->m+1));
    int *h_colInd = (int *) malloc(sizeof(int)*d_mat->nnz);

    checkCudaErrors(cudaMemcpy(h_val, d_mat->d_val, sizeof(double)*d_mat->nnz, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_rowPtr, d_mat->d_rowPtr, sizeof(int)*(d_mat->m+1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_colInd, d_mat->d_colInd, sizeof(int)*d_mat->nnz, cudaMemcpyDeviceToHost));

    initCSR_h(h_mat, d_mat->m, d_mat->n, d_mat->nnz, h_val, h_rowPtr, h_colInd);
    
}


void VEC_h2d(VEC_h *h_v, VEC_d *d_v) {
    double *d_val;
    checkCudaErrors(cudaMalloc((void**)&d_val, sizeof(double)*h_v->n));
    checkCudaErrors(cudaMemcpy(d_val, h_v->h_val, sizeof(double)*h_v->n, cudaMemcpyHostToDevice));
    initVEC_d(d_v, h_v->n, d_val);
}

void VEC_d2h(VEC_d *d_v, VEC_h *h_v) {
    double *h_val = (double *) malloc(sizeof(double) * d_v->n);
    checkCudaErrors(cudaMemcpy(h_val, d_v->d_val, sizeof(double)*d_v->n, cudaMemcpyDeviceToHost));
    initVEC_h(h_v, d_v->n, h_val);
}


void DN_h2d(DN_h *h_mat, DN_d *d_mat) {
    double *d_val;
    checkCudaErrors(cudaMalloc((void**)&d_val, sizeof(double)*h_mat->n*h_mat->m));
    checkCudaErrors(cudaMemcpy(d_val,h_mat->h_val, sizeof(double)*h_mat->n*h_mat->m, cudaMemcpyHostToDevice));
    initDN_d(d_mat, h_mat->m, h_mat->n, d_val);
}

void DN_h2CSR_h(DN_h *mat_dn, CSR_h *mat_csr) {
    int nnz = 0;
    for (int i = 0; i < mat_dn->m; i++) {
        for (int j = 0; j < mat_dn->n; j++) {
            if(mat_dn->h_val[i*mat_dn->n+j] != 0) {
                nnz += 1;
            }
        }
    }

    double *h_val;
    int *h_rowPtr;
    int *h_colInd;

    if(nnz > 0) {
        h_val = (double *) malloc(sizeof(double) *nnz);
        h_rowPtr = (int *) malloc(sizeof(int) * (mat_dn->m + 1));
        h_colInd = (int *) malloc(sizeof(double) *nnz);
        int count = 0;
        h_rowPtr[0] = 0;

        for (int i = 0; i < mat_dn->m; i++) {
            for (int j = 0; j < mat_dn->n; j++) {
                if(mat_dn->h_val[i*mat_dn->n+j] != 0) {
                    h_val[count] = mat_dn->h_val[i*mat_dn->n+j];
                    h_colInd[count] = j;
                    count += 1;
                }
                h_rowPtr[i+1] = count;
            }
        }
    }
    else {
        h_val = NULL;
        h_rowPtr = NULL;
        h_colInd = NULL;
        // h_rowPtr = (int *) malloc(sizeof(int) * (mat_dn->m + 1));
        // memset(h_rowPtr, 0, sizeof(int) * (mat_dn->m + 1));
    }
    initCSR_h(mat_csr, mat_dn->m, mat_dn->n, nnz, h_val, h_rowPtr, h_colInd);
}

void CSR_h2DN_h(CSR_h *mat_csr, DN_h *mat_dn) {
    double *h_val = (double *) malloc(sizeof(double) * mat_csr->m * mat_csr->n);
    memset(h_val, 0, sizeof(double) * mat_csr->m * mat_csr->n);
    int rowIdx = 0;
    int colIdx = 0;
    int rowPtrIdx = 0;
    int nnz_row = mat_csr->h_rowPtr[rowPtrIdx+1] - mat_csr->h_rowPtr[rowPtrIdx];

    for (int i = 0; i < mat_csr->nnz; i++) {
        colIdx = mat_csr->h_colInd[i];

        while (nnz_row <= 0) {
            rowIdx += 1;
            rowPtrIdx += 1;
            nnz_row = mat_csr->h_rowPtr[rowPtrIdx+1] - mat_csr->h_rowPtr[rowPtrIdx];
        }
        
        h_val[rowIdx * mat_csr->n + colIdx] = mat_csr->h_val[i];
        nnz_row -= 1;
    }

    initDN_h(mat_dn, mat_csr->m, mat_csr->n, h_val);
}


/* ------------------------------- Cublas functions -------------------------------*/

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


/* ------------------------------- Cusparse functions -------------------------------*/

// vev C = mat A * vec B
void matMulVec(CSR_d *d_A, VEC_d *d_B, VEC_d *d_C) {
    double alpha = 1;
    double beta = 1;
    checkCudaErrors(cudaMemset(d_C->d_val, 0.0, sizeof(double)*d_C->n));
    
    //prepare buffer
    char *buffer;
    size_t bufferSizeInBytes;

    checkCusparseErrors(cusparseSpMV_bufferSize(cusparseHandle, 
                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 &alpha,
                 d_A->descr,
                 d_B->descr,
                 &beta,
                 d_C->descr,
                 CUDA_R_64F,
                 CUSPARSE_MV_ALG_DEFAULT,
                 &bufferSizeInBytes
                ));

    checkCudaErrors(cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes));

    checkCusparseErrors(cusparseSpMV(cusparseHandle, 
                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 &alpha,
                 d_A->descr,
                 d_B->descr,
                 &beta,
                 d_C->descr,
                 CUDA_R_64F,
                 CUSPARSE_MV_ALG_DEFAULT,
                 buffer
                ));

    cudaFree(buffer);
}

// vec C = vec B * mat A = mat AT * B
void vecMulMat(VEC_d *d_B, CSR_d *d_A, VEC_d *d_C) {
    double alpha = 1;
    double beta = 1;
    checkCudaErrors(cudaMemset(d_C->d_val, 0.0, sizeof(double)*d_C->n));
    
    //prepare buffer
    char *buffer;
    size_t bufferSizeInBytes;

    checkCusparseErrors(cusparseSpMV_bufferSize(cusparseHandle, 
                 CUSPARSE_OPERATION_TRANSPOSE, 
                 &alpha,
                 d_A->descr,
                 d_B->descr,
                 &beta,
                 d_C->descr,
                 CUDA_R_64F,
                 CUSPARSE_MV_ALG_DEFAULT,
                 &bufferSizeInBytes
                ));

    checkCudaErrors(cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes));

    checkCusparseErrors(cusparseSpMV(cusparseHandle, 
                 CUSPARSE_OPERATION_TRANSPOSE, 
                 &alpha,
                 d_A->descr,
                 d_B->descr,
                 &beta,
                 d_C->descr,
                 CUDA_R_64F,
                 CUSPARSE_MV_ALG_DEFAULT,
                 buffer
                ));

    cudaFree(buffer);
}

// mat C = mat A + mat B
void matAdd(CSR_d *d_A, CSR_d *d_B, CSR_d *d_C) {
    double alpha = 1;
    double beta = 1;
    int baseC, nnzC;
    int *nnzTotalDevHostPtr = &nnzC;

    //prepare buffer
    char *buffer;
    size_t bufferSizeInBytes;

    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
    cudaMalloc((void**)&d_C->d_rowPtr, sizeof(int)*(d_A->m+1));

    cusparseMatDescr_t mat_descr;
    cusparseCreateMatDescr(&mat_descr);

    checkCusparseErrors(cusparseDcsrgeam2_bufferSizeExt(cusparseHandle,
                                                        d_A->m,
                                                        d_A->n,
                                                        &alpha,
                                                        mat_descr,
                                                        d_A->nnz,
                                                        d_A->d_val,
                                                        d_A->d_rowPtr,
                                                        d_A->d_colInd,
                                                        &beta,
                                                        mat_descr,
                                                        d_B->nnz,
                                                        d_B->d_val,
                                                        d_B->d_rowPtr,
                                                        d_B->d_colInd,
                                                        mat_descr,
                                                        d_C->d_val,
                                                        d_C->d_rowPtr,
                                                        d_C->d_colInd,
                                                        &bufferSizeInBytes
                        ));
    checkCudaErrors(cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes));

    checkCusparseErrors(cusparseXcsrgeam2Nnz(cusparseHandle, 
                         d_A->m,
                         d_A->n,
                         mat_descr,
                         d_A->nnz,
                         d_A->d_rowPtr,
                         d_A->d_colInd,
                         mat_descr,
                         d_B->nnz,
                         d_B->d_rowPtr,
                         d_B->d_colInd,
                         mat_descr,
                         d_C->d_rowPtr,
                         nnzTotalDevHostPtr,
                         buffer));

    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;

    }else{
        checkCudaErrors(cudaMemcpy(&nnzC, d_C->d_rowPtr+d_A->m, sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(&baseC, d_C->d_rowPtr, sizeof(int), cudaMemcpyDeviceToHost));
        nnzC -= baseC;
    }

    checkCudaErrors(cudaMalloc((void**)&d_C->d_colInd, sizeof(int)*nnzC));
    checkCudaErrors(cudaMalloc((void**)&d_C->d_val, sizeof(double)*nnzC));

    checkCusparseErrors(cusparseDcsrgeam2(cusparseHandle,
                                            d_A->m,
                                            d_A->n,
                                            &alpha,
                                            mat_descr,
                                            d_A->nnz,
                                            d_A->d_val,
                                            d_A->d_rowPtr,
                                            d_A->d_colInd,
                                            &beta,
                                            mat_descr,
                                            d_B->nnz,
                                            d_B->d_val,
                                            d_B->d_rowPtr,
                                            d_B->d_colInd,
                                            mat_descr,
                                            d_C->d_val,
                                            d_C->d_rowPtr,
                                            d_C->d_colInd,
                                            buffer
                        ));
    initCSR_d(d_C, d_A->m, d_A->n, nnzC, d_C->d_val, d_C->d_rowPtr, d_C->d_colInd);

    cudaFree(buffer);
}




void get_input(int n, int m, double P[n][n], double Q[n], double l[m], double u[m], double A[m][n], double AT[n][m]){


};

//// calculate epsilon for termination
//double calc_eps(){
//
//}

// calculate inverse of a diagonal matrix A and store it in matrix B
void inverseDiag(int n, double A[n][n], double B[n][n]){

}

// calculate transpose of a matrix A and store it in matrix B
void transpose(int n, int m, double A[n][m], double B[m][n]){

}

// fill a diagonal n*n matrix A with val on each diagonal entry
void initializeDiagMat(int n, double val, double A[n][n]){

}

// calculate A*B and store it in C
void matMulMat(int n, int m, int t, double A[n][m], double B[m][t], double C[n][t]){

}

// calculate A*B and store it in C, B is diagonal
void matMulDiagMat(int n, int m, int t, double A[n][m], double B[m][t], double C[n][t]){

}

// calculate A*B and store it in C, A is diagonal
void diagMatMulMat(int n, int m, int t, double A[n][m], double B[m][t], double C[n][t]){

}

// calculate matrix A* vector B and store it in vector C, A is diagonal
void diagMatMulVec(int n, int m, double A[n][m], double B[m], double C[n]){

}


// calculate matrix A* vector B and store it in vector C
void matMulVec(int n, int m, double A[n][m], double B[m], double C[n]){

}

// calculate vector A* matrix B and store it in vector C
void vecMulMat(int n, int m, double A[n], double B[n][m], double C[n]){

}

// calculate vector A* matrix B and store it in vector C
void vecMulDiagMat(int n, int m, double A[n], double B[n][m], double C[n]){

}


// calculate sum of two matrices
void matAdd(int n, int m, double A[n][m], double B[n][m], int C[n][m]){

}


// min(l) max(u) projection of an vector
void vecMinMaxProj(int n, double A[n], double l[n], double u[n], double B[n]){


}

// B = val * A
void scalarMulDiagMat(int n, double val, double A[n][n], double B[n][n])
{


}

// calculate norm of a vector
double norm2(int n, double A[n]){
    double norm = 0.0;
    return norm;
}

double normInf(int n, double A[n]){
    double norm = 0.0;
    return norm;
}


void calculateR(int n, double R[n][n], double l[n], double u[n], double rho){

}

//get diagonal element of K into M
void calculatePrecond(int n, double K[n][n], double M[n][n]){

}


// calculate if the program should terminate
bool termination(int n, int m, double x[n], double y[m], double z[m], double P[n][n], double Q[n], double A[m][n], double AT[n][m], double epsilon){
    // temp1 = A*x
    double temp1[m], temp2[m], residualPrimal[m];
    //calculate residual of Primal
    matMulVec(m, n, A, x, temp1); // temp1 = A*x;
    scalarMulVec(m, -1, z, temp2); // temp2 = -z;
    vecAdd(m, temp1, temp2, residualPrimal); // residualPrimal = A*x - z;

    //calculate residual of Dual
    double temp3[n], temp4[n], temp5[n], residualDual[n];
    matMulVec(n, n, P, x, temp3); // temp3 = P*x;
    matMulVec(n, m, AT, y, temp4); // temp4 = A^T * y;
    vecAdd(n, temp3, temp4, temp5); // temp5 = temp3 + temp4;
    vecAdd(n, temp5, Q, residualDual); // residualDual = temp5 + Q;
    if ((norm2(residualPrimal) <= epsilon) && (norm2(residualDual) <= epsilon))
        return 1;
    else
        return 0;
}


// algorithm 3 solve kkt with PCG
void solveKKT(int n, int m, double x[n], double y[m], double z[m], double P[n][n], double Q[n], double A[m][n], double AT[n][m], double l[m], double u[m], double R[m][m], double xNext[n], double zNext[m], double rho, double sigma, double epsilon){
    //todo Add a function to update rho in every 10 iterations
    double K[n][n], double I[n][n];

    //calculate K = P + sigmaI + A^T*R*A, K is symmetric positive definite matrix;
    // initialize an identity matrix I
    initializeDiagMat(n, I, 1);
    double temp1[n][m], temp2[n][n], temp3[n][n], temp4[n][n];
    matMulDiagMat(n,m,m,AT, R, temp1); // temp1 = A^T * R;
    matMulMat(n,m,n, temp1, A, temp2); // temp2 = A^T * R * A;
    scalarMulDiagMat(n, sigma, I, temp3); // temp3 = sigma * I;
    matAdd(n, n, temp2, temp3, temp4);
    matAdd(n, n, temp4, P, K);  // K = P + sigma * I + A^T * R * A

    //calculate preconditioner
    double M[n][n], MINV[n][n];
    calculatePrecond(n, K, M);
    inverseDiag(m, M, MINV);
    //calculate b = sigma * x - q + A^T(Rz-y)
    double b[n];
    double temp5[m], temp6[m], temp7[m], temp8[n], temp9[n], temp10[n], temp11[n];
    diagMatMulVec(m, m, R, z, temp5);
    scalarMulVec(m, -1, y, temp6);
    vecAdd(m, temp5, temp6, temp7); // temp7 = Rz - y;
    matMulVec(n, m, AT, temp7, temp8); // temp8 = A^T(Rz - y);
    scalarMulVec(n, sigma, x, temp9); // temp9 = sigma * x;
    scalarMulVec(n, -1, Q, temp10); // temp10 = -q;
    vecAdd(n, temp9, temp10, temp11); // temp11 = sigma * x - q;
    vecAdd(n, temp11, temp8, b); // b = sigma * x - q + A^T(Rz - y);

    //initialize r0=Kx-b, y0, p0
    double r[n];
    matMulVec(n, n, K, x, temp8); // temp8 = Kx;
    scalarMulVec(n, -1, b, temp9); // temp9 = -b;
    vecAdd(n, temp8, temp9, r);

    //intitialize y0 = M^(-1)*r;
    double y_kkt[n];
    diagMatMulVec(n,n, MINV, r, y_kkt); // y0 = M^(-1)*r

    // intitialize p0 = -y0;
    double p[n];
    scalarMulVec(n, -1, y_kkt, p);
    int k = 0;

    while (norm2(n,r) > epsilon * norm2(n,b))
    {
        //calculate a^k
        double alpha;
        double temp12, temp13;
        temp12 = innerProduct(n, r, y_kkt); // calculate Numerator r^T * y;
        vecMulMat(n,n, p, K, temp8); // temp8 = p^T * K
        temp13 = innerProduct(n,temp8, p); // calculate denominator p^T * K * p;
        alpha = temp12 / temp13;
        //calculate x^k+1
        scalarMulVec(n, alpha, p, temp8);
        if (k==0)
            vecAdd(n, x, temp8, xNext); // x = x + alpha * p;
        else
            vecAdd(n, xNext, temp8, xNext);
        //calculate r^k+1
        double r[n];
        diagMatMulVec(n, n, K, p,temp8); // temp8 = K * p;
        scalarMulVec(n,alpha, temp8, temp9); // temp9 = alpha * K * p;
        vecAdd(n, r, temp9, r);
        //calculate y^k+1
        double yNew[n];
        diagMatMulVec(n, n, MINV, r, y_kkt); // y = M^(-1) * rNew;
        //calculate beta^k+1
        double beta;
        temp13 = innerProduct(n, r, y_kkt); // calculate Numerator rNew^T * yNew;
        beta = temp13 / temp12; // beta = ( rNew^T * yNew) / ( r^T * y)
        //calculate p^k+1
        scalarMulVec(n, -1, y_kkt, temp8); // temp8 = -y
        scalarMulVec(n, beta, p, temp9); // temp9 = beta * p;
        vecAdd(n, temp8, temp9, p);
        k+=1;
    }
    matMulVec(m,n, A, xNext, zNext);
}



int main() {
    //
    int n,m;
    //get n and m
    // input
    double P[n][n],Q[n], A[m][n], AT[n][m], l[m],u[m];
    get_input(P,Q,l,u,A);

    //get AT = A^T
    transpose(m,n, A, AT);
    //initialize x,y,z
    double x[n], y[m], z[m];
    memset(x, 0, sizeof(x));
    memset(y, 0, sizeof(y));
    memset(z, 0, sizeof(z));
    // initialize sigma and alpha
    double sigma = 0.000001, alpha=1.6, rho = 0.5;
    // initialize epsilon, now we assume epsilon is a constant
    // todo add a function to calculate epsilon in each iteration;
    double epsilon = 0.00001;
    //eps = calc_eps();
    int k = 0;
    while (!termination(n, m, x, y, z, P, Q, A, AT, epsilon))
    {
        double xNext[n], zNext[n];
        double R[m][m], RINV[m][m];

        //calculate the penalty matrix R and its inverse RINV
        calculateR(R);
        inverseDiag(m, R, RINV);

        solveKKT(n, m, x, y, z, P, Q, A, AT, l,u, R, xNext, zNext, rho, sigma, epsilon);
        // update x
        double temp1[n], temp2[n]
        scalarMulVec(n, alpha, xNext, temp1); // temp1 = alpha * xNext;
        scalarMulVec(n, 1 - alpha, x, temp2); // temp2 = (1 - alpha) * x;
        vecAdd(n, temp1, temp2, x); // x = alpha * xNext +  (1 - alpha) * x;

        // update z
        double temp3[m], temp4[m], temp5[m], temp6[m], temp7[m], zNextReal;
        scalarMulVec(m, alpha, zNext, temp3); // temp3 = alpha * zNext;
        scalarMulVec(m, 1 - alpha, z, temp4); // temp4 = (1 - alpha) * z;
        matMulVec(m,m, RINV,y, temp5 ); // temp5 = R^(-1) * y;
        vecAdd(m, temp3, temp4, temp6);  // temp6 = alpha * zNext + (1 - alpha) * z;
        vecAdd(m, temp5, temp6, temp7);    // temp7 = alpha * zNext + (1 - alpha) * z + R^(-1) * y
        vecMinMaxProj(m, temp7, l, u, zNextReal); // zNextReal is the projection of temp7

        // update y
        scalarMulVec(m, -1, zNextReal, temp3); // temp3 = -zNextReal;
        vecAdd(m, temp6, temp3, temp4); // temp4 = alpha * zNext + (1 - alpha) * z - zNextReal;
        matMulVec(m, m, R, temp4, temp5); // temp5 = R * (alpha * zNext + (1 - alpha) * z - zNextReal);
        vecAdd(m, y, temp5, y); // y = y + temp5;

        // update z^k to z^(k+1)
        z = zNextReal; // z = zNextReal
        k += 1;
    }
    return 0;
}
