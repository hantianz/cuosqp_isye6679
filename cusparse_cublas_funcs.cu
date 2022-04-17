// nvcc cusparse.cu  -lcusparse -lcublas -o  cusparse
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "support.h"
#include "cublas_v2.h"
#include <cusparse.h>

void checkCudaErrors(cudaError_t cuda_ret) {
    if(cuda_ret != cudaSuccess) {
        printf("CUDA Error: %s", cudaGetErrorString (cuda_ret));
    }
}
void checkCublasErrors(cublasStatus_t cuda_ret) {
    if(cuda_ret != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error: %d", cuda_ret);
    }
}

void checkCusparseErrors(cusparseStatus_t cuda_ret) {
    if(cuda_ret != CUSPARSE_STATUS_SUCCESS) {
        printf("Cusparse Error: %s", cusparseGetErrorString(cuda_ret));
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

void copyCSR_h(CSR_h *mat_src, CSR_h *mat_dest) {
    double *h_val = (double*) malloc(sizeof(double)*mat_src->nnz);
    int *h_rowPtr = (int *) malloc(sizeof(int)*(mat_src->m+1));
    int *h_colInd = (int *) malloc(sizeof(int)*mat_src->nnz);

    memcpy(h_val, mat_src->h_val, sizeof(double)*mat_src->nnz);
    memcpy(h_rowPtr, mat_src->h_rowPtr, sizeof(int)*(mat_src->m+1));
    memcpy(h_colInd, mat_src->h_colInd, sizeof(int)*mat_src->nnz);

    initCSR_h(mat_dest, mat_src->m, mat_src->n, mat_src->nnz, h_val, h_rowPtr, h_colInd);
}

void copyCSR_d(CSR_d *mat_src, CSR_d *mat_dest) {
    double *d_val;
    int *d_rowPtr;
    int *d_colInd;

    checkCudaErrors(cudaMalloc((void**)&d_val, sizeof(double)*mat_src->nnz));
    checkCudaErrors(cudaMalloc((void**)&d_rowPtr, sizeof(int)*(mat_src->m+1)));
    checkCudaErrors(cudaMalloc((void**)&d_colInd, sizeof(int)*mat_src->nnz));

    checkCudaErrors(cudaMemcpy(d_val, mat_src->d_val, sizeof(double)*mat_src->nnz, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_rowPtr, mat_src->d_rowPtr, sizeof(int)*(mat_src->m+1), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_colInd, mat_src->d_colInd, sizeof(int)*mat_src->nnz, cudaMemcpyDeviceToDevice));    

    initCSR_d(mat_dest, mat_src->m, mat_src->n, mat_src->nnz, d_val, d_rowPtr, d_colInd);
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

void copyVEC_h(VEC_h *vec_src, VEC_h *vec_dest) {
    double *h_val = (double*) malloc(sizeof(double)*vec_src->n);
    memcpy(h_val, vec_src->h_val, sizeof(double)*vec_src->n);
    initVEC_h(vec_dest, vec_src->n, h_val);
}

void copyVEC_d(VEC_d *vec_src, VEC_d *vec_dest) {
    double *d_val;
    checkCudaErrors(cudaMalloc((void**)&d_val, sizeof(double)*vec_src->n));
    checkCudaErrors(cudaMemcpy(d_val, vec_src->d_val, sizeof(double)*vec_src->n, cudaMemcpyDeviceToDevice));
    initVEC_d(vec_dest, vec_src->n, d_val);
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

void copyDN_h(DN_h *mat_src, DN_h *mat_dest) {
    double *h_val = (double*) malloc(sizeof(double)*(mat_src->n * mat_src->m));
    memcpy(h_val, mat_src->h_val, sizeof(double)*(mat_src->n * mat_src->m));
    initDN_h(mat_dest, mat_src->m, mat_src->n, h_val);
}

void copyDN_d(DN_d *mat_src, DN_d *mat_dest) {
    double *d_val;
    checkCudaErrors(cudaMalloc((void**)&d_val, sizeof(double)*(mat_src->n * mat_src->m)));
    checkCudaErrors(cudaMemcpy(d_val, mat_src->d_val, sizeof(double)*(mat_src->n * mat_src->m), cudaMemcpyDeviceToDevice));
    initDN_d(mat_dest, mat_src->m, mat_src->n, d_val);
}

/* ------------------------------- destructor -------------------------------*/
void destroyVEC_h(VEC_h *v) {
    free(v->h_val);
    free(v);
}

void destroyVEC_d(VEC_d *v) {
    cusparseDestroyDnVec(v->descr);
    cudaFree(v->d_val);
    free(v);
}

void destroyDN_h(DN_h *mat) {
    free(mat->h_val);
    free(mat);
}

void destroyDN_d(DN_d *mat) {
    cusparseDestroyDnMat(mat->descr);
    cudaFree(mat->d_val);
    free(mat);
}

void destroyCSR_h(CSR_h *mat) {
    free(mat->h_val);
    free(mat->h_colInd);
    free(mat->h_rowPtr);
    free(mat);
}

void destroyCSR_d(CSR_d *mat) {
    cusparseDestroySpMat(mat->descr);
    cudaFree(mat->d_val);
    cudaFree(mat->d_rowPtr);
    cudaFree(mat->d_colInd);
    free(mat);
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
void vecAdd(VEC_d *d_A, VEC_d *d_B, VEC_d *d_C) {
    double alpha = 1;
    d_C->n = d_A->n;
    checkCudaErrors(cudaMalloc((void**)&d_C->d_val, sizeof(double)*d_A->n));
    checkCublasErrors(cublasDcopy(cublasHandle, d_A->n, d_A->d_val, 1, d_C->d_val, 1));
    checkCublasErrors(cublasDaxpy(cublasHandle, d_A->n, &alpha, d_B->d_val, 1, d_C->d_val, 1));
    checkCusparseErrors(cusparseCreateDnVec(&d_C->descr, d_C->n, d_C->d_val, CUDA_R_64F));
}

// A = A + B
void vecAddInPlace(VEC_d *d_A, VEC_d *d_B) {
    double alpha = 1;
    checkCublasErrors(cublasDaxpy(cublasHandle, d_A->n, &alpha, d_B->d_val, 1, d_A->d_val, 1));
    checkCusparseErrors(cusparseCreateDnVec(&d_A->descr, d_A->n, d_A->d_val, CUDA_R_64F));
}


// return inner project of two vectors
// c = A * B
double innerProduct(VEC_d *d_A, VEC_d *d_B) {
    double h_C;
    checkCublasErrors(cublasDdot(cublasHandle, d_A->n, d_A->d_val, 1, d_B->d_val, 1, &h_C));
    return h_C;
}

// B = val * A
void scalarMulVec(double sc, VEC_d *d_A, VEC_d *d_B) {
    d_B->n = d_A->n;
    checkCudaErrors(cudaMalloc((void**)&d_B->d_val, sizeof(double)*d_A->n));
    checkCudaErrors(cudaMemset(d_B->d_val, 0.0, sizeof(double)*d_A->n));
    checkCublasErrors(cublasDaxpy(cublasHandle, d_A->n, &sc, d_A->d_val, 1, d_B->d_val, 1));
    checkCusparseErrors(cusparseCreateDnVec(&d_B->descr, d_B->n, d_B->d_val, CUDA_R_64F));
}

// scale a vector in place
// A = val * A
void scaleMulVecInPlace(double sc, VEC_d *d_A) {
    checkCublasErrors(cublasDscal(cublasHandle, d_A->n, &sc, d_A->d_val, 1));
    checkCusparseErrors(cusparseCreateDnVec(&d_A->descr, d_A->n, d_A->d_val, CUDA_R_64F));
}


/* ------------------------------- Cusparse functions -------------------------------*/
// vev C = mat A * vec B
void matMulVec(CSR_d *d_A, VEC_d *d_B, VEC_d *d_C) {
    double alpha = 1;
    double beta = 1;

    // init d_C
    d_C->n = d_A->m;
    checkCudaErrors(cudaMalloc((void**)&d_C->d_val, sizeof(double)*d_C->n));
    checkCudaErrors(cudaMemset(d_C->d_val, 0.0, sizeof(double)*d_C->n));
    checkCusparseErrors(cusparseCreateDnVec(&d_C->descr, d_C->n, d_C->d_val, CUDA_R_64F));
    
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

    // init d_C
    d_C->n = d_A->n;
    checkCudaErrors(cudaMalloc((void**)&d_C->d_val, sizeof(double)*d_C->n));
    checkCudaErrors(cudaMemset(d_C->d_val, 0.0, sizeof(double)*d_C->n));
    checkCusparseErrors(cusparseCreateDnVec(&d_C->descr, d_C->n, d_C->d_val, CUDA_R_64F));

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

    cusparseDestroyMatDescr(mat_descr);

    cudaFree(buffer);
}

// mat C = mat A * mat B
void matMulMat(CSR_d *d_A, CSR_d *d_B, CSR_d *d_C) {
    int baseC, nnzC;
    void *buffer = NULL;
    size_t bufferSizeInBytes;

    int *nnzTotalDevHostPtr = &nnzC;

    double alpha = 1;

    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);

    cusparseMatDescr_t mat_descr;
    cusparseCreateMatDescr(&mat_descr);

    csrgemm2Info_t info = NULL;
    cusparseCreateCsrgemm2Info(&info);

    //prepare buffer
    checkCusparseErrors(cusparseDcsrgemm2_bufferSizeExt(cusparseHandle,
                                                    d_A->m,
                                                    d_B->n,
                                                    d_A->n,
                                                    &alpha,
                                                    mat_descr,
                                                    d_A->nnz,
                                                    d_A->d_rowPtr,
                                                    d_A->d_colInd,
                                                    mat_descr,
                                                    d_B->nnz,
                                                    d_B->d_rowPtr,
                                                    d_B->d_colInd,
                                                    NULL,
                                                    mat_descr,
                                                    d_A->nnz,
                                                    d_A->d_rowPtr,
                                                    d_A->d_colInd,
                                                    info,
                                                    &bufferSizeInBytes
                                                    ));
    checkCudaErrors(cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes));

    cudaMalloc((void**)&d_C->d_rowPtr, sizeof(int)*(d_A->m+1));

    checkCusparseErrors(cusparseXcsrgemm2Nnz(cusparseHandle,
                                         d_A->m,
                                         d_B->n,
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
                                         d_A->nnz,
                                         d_A->d_rowPtr,
                                         d_A->d_colInd,
                                         mat_descr,
                                         d_C->d_rowPtr,
                                         nnzTotalDevHostPtr,
                                         info,
                                         buffer
                                         ));

    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzC, d_C->d_rowPtr+d_A->m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, d_C->d_rowPtr, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }

    cudaMalloc((void**)&d_C->d_colInd, sizeof(int)*nnzC);
    cudaMalloc((void**)&d_C->d_val, sizeof(double)*nnzC);

    // Remark: set csrValC to null if only sparsity pattern is required.
    checkCusparseErrors(cusparseDcsrgemm2(cusparseHandle,
                                        d_A->m,
                                        d_B->n,
                                        d_A->n,
                                        &alpha,
                                        mat_descr,
                                        d_A->nnz,
                                        d_A->d_val,
                                        d_A->d_rowPtr,
                                        d_A->d_colInd,
                                        mat_descr,
                                        d_B->nnz,
                                        d_B->d_val,
                                        d_B->d_rowPtr,
                                        d_B->d_colInd,
                                        NULL,
                                        mat_descr,
                                        d_A->nnz,
                                        d_A->d_val,
                                        d_A->d_rowPtr,
                                        d_A->d_colInd,
                                        mat_descr,
                                        d_C->d_val,
                                        d_C->d_rowPtr,
                                        d_C->d_colInd,
                                        info,
                                        buffer
                    ));

    initCSR_d(d_C, d_A->m, d_B->n, nnzC, d_C->d_val, d_C->d_rowPtr, d_C->d_colInd);

    cusparseDestroyCsrgemm2Info(info);
    cusparseDestroyMatDescr(mat_descr);
                                    
    cudaFree(buffer);
}

void scalarMulMat(double sc, CSR_d *d_A, CSR_d *d_C) {
    int baseC, nnzC;
    void *buffer = NULL;
    size_t bufferSizeInBytes;

    int *nnzTotalDevHostPtr = &nnzC;

    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);

    cusparseMatDescr_t mat_descr;
    cusparseCreateMatDescr(&mat_descr);

    csrgemm2Info_t info = NULL;
    cusparseCreateCsrgemm2Info(&info);

    //prepare buffer
    checkCusparseErrors(cusparseDcsrgemm2_bufferSizeExt(cusparseHandle,
                                                    d_A->m,
                                                    d_A->n,
                                                    0,
                                                    NULL,
                                                    mat_descr,
                                                    d_A->nnz,
                                                    d_A->d_rowPtr,
                                                    d_A->d_colInd,
                                                    mat_descr,
                                                    d_A->nnz,
                                                    d_A->d_rowPtr,
                                                    d_A->d_colInd,
                                                    &sc,
                                                    mat_descr,
                                                    d_A->nnz,
                                                    d_A->d_rowPtr,
                                                    d_A->d_colInd,
                                                    info,
                                                    &bufferSizeInBytes
                                                    ));
    checkCudaErrors(cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes));

    cudaMalloc((void**)&d_C->d_rowPtr, sizeof(int)*(d_A->m+1));

    checkCusparseErrors(cusparseXcsrgemm2Nnz(cusparseHandle,
                                         d_A->m,
                                         d_A->n,
                                         0,
                                         mat_descr,
                                         d_A->nnz,
                                         d_A->d_rowPtr,
                                         d_A->d_colInd,
                                         mat_descr,
                                         d_A->nnz,
                                         d_A->d_rowPtr,
                                         d_A->d_colInd,
                                         mat_descr,
                                         d_A->nnz,
                                         d_A->d_rowPtr,
                                         d_A->d_colInd,
                                         mat_descr,
                                         d_C->d_rowPtr,
                                         nnzTotalDevHostPtr,
                                         info,
                                         buffer
                                         ));
    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzC, d_C->d_rowPtr+d_A->m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, d_C->d_rowPtr, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }

    cudaMalloc((void**)&d_C->d_colInd, sizeof(int)*nnzC);
    cudaMalloc((void**)&d_C->d_val, sizeof(double)*nnzC);

    // Remark: set csrValC to null if only sparsity pattern is required.
    checkCusparseErrors(cusparseDcsrgemm2(cusparseHandle,
                                        d_A->m,
                                        d_A->n,
                                        0,
                                        NULL,
                                        mat_descr,
                                        d_A->nnz,
                                        d_A->d_val,
                                        d_A->d_rowPtr,
                                        d_A->d_colInd,
                                        mat_descr,
                                        d_A->nnz,
                                        d_A->d_val,
                                        d_A->d_rowPtr,
                                        d_A->d_colInd,
                                        &sc,
                                        mat_descr,
                                        d_A->nnz,
                                        d_A->d_val,
                                        d_A->d_rowPtr,
                                        d_A->d_colInd,
                                        mat_descr,
                                        d_C->d_val,
                                        d_C->d_rowPtr,
                                        d_C->d_colInd,
                                        info,
                                        buffer
                    ));

    initCSR_d(d_C, d_A->m, d_A->n, nnzC, d_C->d_val, d_C->d_rowPtr, d_C->d_colInd);

    cusparseDestroyCsrgemm2Info(info);
    cusparseDestroyMatDescr(mat_descr);
                                    
    cudaFree(buffer);
}

// B = AT
void transpose(CSR_d *d_A, CSR_d *d_B) {
    char *buffer;
    size_t bufferSizeInBytes;

    // init d_B (n x m)
    checkCudaErrors(cudaMalloc((void**)&d_B->d_val, sizeof(double)*d_A->nnz));
    checkCudaErrors(cudaMalloc((void**)&d_B->d_rowPtr, sizeof(int)*(d_A->n+1)));
    checkCudaErrors(cudaMalloc((void**)&d_B->d_colInd, sizeof(int)*d_A->nnz));
    
    checkCusparseErrors(cusparseCsr2cscEx2_bufferSize(cusparseHandle,
                                                      d_A->m,
                                                      d_A->n,
                                                      d_A->nnz,
                                                      d_A->d_val,
                                                      d_A->d_rowPtr,
                                                      d_A->d_colInd,
                                                      d_B->d_val,
                                                      d_B->d_rowPtr,
                                                      d_B->d_colInd,
                                                      CUDA_R_64F,
                                                      CUSPARSE_ACTION_NUMERIC,
                                                      CUSPARSE_INDEX_BASE_ZERO,
                                                      CUSPARSE_CSR2CSC_ALG1,
                                                      &bufferSizeInBytes));

    checkCudaErrors(cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes));

    checkCusparseErrors(cusparseCsr2cscEx2(cusparseHandle,
                                              d_A->m,
                                              d_A->n,
                                              d_A->nnz,
                                              d_A->d_val,
                                              d_A->d_rowPtr,
                                              d_A->d_colInd,
                                              d_B->d_val,
                                              d_B->d_rowPtr,
                                              d_B->d_colInd,
                                              CUDA_R_64F,
                                              CUSPARSE_ACTION_NUMERIC,
                                              CUSPARSE_INDEX_BASE_ZERO,
                                              CUSPARSE_CSR2CSC_ALG1,
                                              buffer));

    initCSR_d(d_B, d_A->n, d_A->m, d_A->nnz, d_B->d_val, d_B->d_rowPtr, d_B->d_colInd);
    cudaFree(buffer);
}

void printMatrix(int m, int n, double *A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f,", A[i*n+j]);
        }
        printf("\n");
    }
}

void printMatrix_d(int m, int n, double *A) {
    double *h_A = (double *) malloc(sizeof(double) * m * n);
    checkCudaErrors(cudaMemcpy(h_A, A, sizeof(double)*n*m, cudaMemcpyDeviceToHost));
    printMatrix(m, n, h_A);
    free(h_A);
}

void printDVec(int n, double *A) {
    for (int i = 0; i < n; i++) {
        printf("%f, ", A[i]);
    }
}

void printIVec(int n, int *A) {
    for (int i = 0; i < n; i++) {
        printf("%d, ", A[i]);
    }
}

void printDVec_d(int n, double *A) {
    double *h_A = (double *) malloc(sizeof(double) * n);
    checkCudaErrors(cudaMemcpy(h_A, A, sizeof(double)*n, cudaMemcpyDeviceToHost));
    printDVec(n, h_A);
    free(h_A);
}

void printIVec_d(int n, int *A) {
    int *h_A = (int *) malloc(sizeof(int) * n);
    checkCudaErrors(cudaMemcpy(h_A, A, sizeof(int)*n, cudaMemcpyDeviceToHost));
    printIVec(n, h_A);
    free(h_A);
}

void printVech(VEC_h *vec) {
    printDVec(vec->n, vec->h_val);
    printf("\n");
}

void printVecd(VEC_d *vec) {
    printDVec_d(vec->n, vec->d_val);
    printf("\n");
}

void printCSRh(CSR_h *mat) {
    DN_h *mat_dn = (DN_h *) malloc(sizeof(DN_h));
    CSR_h2DN_h(mat, mat_dn);
    printMatrix(mat_dn->m, mat_dn->n, mat_dn->h_val);
    printf("\n");
}

void printCSRd(CSR_d *mat) {
    CSR_h *mat_h = (CSR_h *) malloc(sizeof(CSR_h));
    CSR_d2h(mat, mat_h);
    printCSRh(mat_h);
}

