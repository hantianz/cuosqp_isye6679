// nvcc cusparse.cu  -lcusparse -lcublas -o  cusparse
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "support.h"
#include "cublas_v2.h"
#include <cusparse.h>
#include "cusparse_cublas_funcs.cu"
#include <time.h>

double randReal(double low, double high) {
    double d;
    d = (double) rand() / ((double) RAND_MAX + 1);
    d = (low + d * (high - low));
    d = ((int)(d * 1000)) / 1000.0;
    return d;
}

int generateRandomSparseMatrix(int m, int n, double *A, double ratio) {
    int nnz = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if(randReal(0, 1) < ratio) {
               A[i*n+j] = randReal(0, 100);
               nnz += 1;
            }
            else {
               A[i*n+j] = 0;
            }
        }
    }
    return nnz;
}

void generateRandomVector(int n, double *A) {
    for (int i = 0; i < n; i++) {
        A[i] = randReal(0, 100);
    }
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


/* CPU version */

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

void matMulMat_cpu(int n, int m, int t, double *A, double *B, double *C){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < t; ++j) {
            C[i*t+j] = 0;
            for (int k = 0; k < m; ++k)
                C[i*t+j] += A[i*m+k] * B[k*t+j];
        }
}

// calculate vector A* matrix B and store it in vector C
void vecMulMat_cpu(int n, int m, double *A, double *B, double *C) {
    for (int i = 0; i < m; ++i) {
        C[i] = 0;
        for (int j = 0; j < n; ++j)
            C[i] += A[j] * B[j*m+i];
    }
}


// calculate matrix A* vector B and store it in vector C
void matMulVec_cpu(int n, int m, double *A, double *B, double *C){
    for (int i = 0; i < n; ++i)
    {
        C[i] = 0;
        for (int j = 0; j < m; ++j)
            C[i] += A[i*m + j] * B[j];
    }

}

// calculate sum of two matrices
void matAdd_cpu(int n, int m, double *A, double *B, double *C){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i*m+j] = A[i*m+j] + B[i*m+j];
}

void scalarMulMat_cpu(int m, int n, double val, double *A, double *B)
{
    for (int i = 0; i < m * n; ++i)
        B[i] = A[i] * val;
}



void verifyVecAdd(int n, double *A, double *B, double *C_gpu) {
    const float relativeTolerance = 1e-6;
    double C_cpu[n];

    clock_t tic = clock(); 
    vecAdd_cpu(n, A, B, C_cpu);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);

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
    clock_t tic = clock(); 
    double C_cpu = innerProduct_cpu(n, A, B);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);
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

    clock_t tic = clock(); 
    scalarMulVec_cpu(n, val, A, C_cpu);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);
  

  for(int i = 0; i < n; i++) {
    double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
    if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
      printf("TEST FAILED\n");
      exit(0);
    }
  }
  printf("TEST PASSED\n");
}


void verifyMatMulVec(int m, int n, double *A, double *B, double *C_gpu) {
    const float relativeTolerance = 1e-6;
    double C_cpu[m];

    clock_t tic = clock(); 
    matMulVec_cpu(m, n, A, B, C_cpu);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);

      for(int i = 0; i < m; i++) {
        double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
          printf("TEST FAILED\n");
          exit(0);
        }
      }
      printf("TEST PASSED\n");
}

void verifyVecMulMat(int m, int n, double *A, double *B, double *C_gpu) {
    // B[1, m] x A[m, n] = C[1 , n]
    const float relativeTolerance = 1e-6;
    double C_cpu[n];

    clock_t tic = clock(); 
    vecMulMat_cpu(m, n, B, A, C_cpu);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);

      for(int i = 0; i < n; i++) {
        double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
          printf("TEST FAILED\n");
          exit(0);
        }
      }
      printf("TEST PASSED\n");
}

void verifyMatAdd(int m, int n, double *A, double *B, double *C_gpu) {
    // B[1, m] x A[m, n] = C[1 , n]
    const float relativeTolerance = 1e-6;
    double C_cpu[m*n];

    clock_t tic = clock(); 
    matAdd_cpu(m, n, A, B, C_cpu);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);

      for(int i = 0; i < m*n; i++) {
        double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
          printf("TEST FAILED\n");
          exit(0);
        }
      }
      printf("TEST PASSED\n");
}

void verifyMatMulMat(int m, int n, int k, double *A, double *B, double *C_gpu) {
    // B[1, m] x A[m, n] = C[1 , n]
    const float relativeTolerance = 1e-6;
    double C_cpu[m*k];

    clock_t tic = clock(); 
    matMulMat_cpu(m, n, k, A, B, C_cpu);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);

    for(int i = 0; i < m*k; i++) {
        double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
          printf("TEST FAILED\n");
          exit(0);
         }
    }
    printf("TEST PASSED\n");
}


void verifyScalarMulMat(int m, int n, double val, double *A, double* C_gpu) {
    const float relativeTolerance = 1e-6;
    double C_cpu[m*n];

    clock_t tic = clock(); 
    scalarMulMat_cpu(m, n, val, A, C_cpu);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("CPU time %fs, ", time_used);
  

    for(int i = 0; i < m*n; i++) {
        double relativeError = (C_cpu[i] - C_gpu[i])/ C_cpu[i];
        if (relativeError > relativeTolerance || relativeError < -relativeTolerance) {
          printf("TEST FAILED\n");
          exit(0);
        }
    }
  printf("TEST PASSED\n");
}



void testVecAdd(int n) {
    printf("- TEST vecAdd\n");
    double* h_A_val = (double*) malloc( sizeof(double)*n );
    double* h_B_val = (double*) malloc( sizeof(double)*n );
    generateRandomVector(n, h_A_val);
    generateRandomVector(n, h_B_val);

    // init vec with value
    VEC_h *h_A = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_A, n, h_A_val);

    VEC_h *h_B = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_B, n, h_B_val);

    // move to gpu vec_d
    VEC_d *d_A = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_A, d_A);

    VEC_d *d_B = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_B, d_B);

    //call function
    VEC_d *d_C = (VEC_d *) malloc(sizeof(VEC_d)); //no initialization

    clock_t tic = clock(); 
    vecAdd(d_A, d_B, d_C);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    // move to cpu
    VEC_h *h_C = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_d2h(d_C, h_C);
    verifyVecAdd(n, h_A->h_val, h_B->h_val, h_C->h_val);

    destroyVEC_d(d_A);
    destroyVEC_d(d_B);
    destroyVEC_d(d_C);
    destroyVEC_h(h_A);
    destroyVEC_h(h_B);
    destroyVEC_h(h_C);
}

void testVecAddInPlace(int n) {
    printf("- TEST vecAddInPlace\n");
    double* h_A_val = (double*) malloc( sizeof(double)*n );
    double* h_B_val = (double*) malloc( sizeof(double)*n );
    generateRandomVector(n, h_A_val);
    generateRandomVector(n, h_B_val);

    // init vec with value
    VEC_h *h_A = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_A, n, h_A_val);

    VEC_h *h_B = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_B, n, h_B_val);

    VEC_h *h_old_A = (VEC_h *) malloc(sizeof(VEC_h));
    copyVEC_h(h_A, h_old_A);

    // move to gpu vec_d
    VEC_d *d_A = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_A, d_A);

    VEC_d *d_B = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_B, d_B);

    clock_t tic = clock(); 
    vecAddInPlace(d_A, d_B);
    clock_t toc = clock();

    VEC_h *h_new_A = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_d2h(d_A, h_new_A);

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    // move to cpu
    verifyVecAdd(n, h_old_A->h_val, h_B->h_val, h_new_A->h_val);

    destroyVEC_d(d_A);
    destroyVEC_d(d_B);
    destroyVEC_h(h_A);
    destroyVEC_h(h_B);
    destroyVEC_h(h_old_A);
}

void testInnerProduct(int n) {
    printf("- TEST innerProduct\n");
    double* h_A_val = (double*) malloc( sizeof(double)*n );
    double* h_B_val = (double*) malloc( sizeof(double)*n );
    generateRandomVector(n, h_A_val);
    generateRandomVector(n, h_B_val);

    // init vec with value
    VEC_h *h_A = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_A, n, h_A_val);

    VEC_h *h_B = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_B, n, h_B_val);

    // move to gpu vec_d
    VEC_d *d_A = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_A, d_A);

    VEC_d *d_B = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_B, d_B);

    //call function
    clock_t tic = clock(); 
    double C = innerProduct(d_A, d_B);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    verifyInnerProduct(n, h_A->h_val, h_B->h_val, C);

    destroyVEC_d(d_A);
    destroyVEC_d(d_B);
    destroyVEC_h(h_A);
    destroyVEC_h(h_B);
}

void testScalarMulVec(int n, double sc) {
    printf("- TEST scalarMulVec\n");
    double* h_A_val = (double*) malloc( sizeof(double)*n );
    double* h_B_val = (double*) malloc( sizeof(double)*n );
    generateRandomVector(n, h_A_val);
    generateRandomVector(n, h_B_val);

    // init vec with value
    VEC_h *h_A = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_A, n, h_A_val);

    // move to gpu vec_d
    VEC_d *d_A = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_A, d_A);

    //call function
    VEC_d *d_B = (VEC_d *) malloc(sizeof(VEC_d));

    clock_t tic = clock(); 
    scalarMulVec(sc, d_A, d_B);
    clock_t toc = clock();

    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    VEC_h *h_B = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_d2h(d_B, h_B);

    verifyScaleMulVec(n, sc, h_A->h_val, h_B->h_val);
    destroyVEC_d(d_A);
    destroyVEC_d(d_B);
    destroyVEC_h(h_A);
    destroyVEC_h(h_B);
}

void testMatMulVec(int m, int n) {
    printf("- TEST matMulVec\n");

    double* h_A_val = (double *) malloc(sizeof(double)*n*m);
    double* h_B_val = (double *) malloc(sizeof(double)*n);

    generateRandomSparseMatrix(m, n, h_A_val, 0.2);
    generateRandomVector(n, h_B_val);
    
    // init csr and vec
    DN_h *h_A_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_A_dn, m, n, h_A_val);

    CSR_h *h_A = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_A_dn, h_A);

    VEC_h *h_B = (VEC_h*) malloc(sizeof(VEC_h));
    initVEC_h(h_B, n, h_B_val);

    // move to gpu
    CSR_d *d_A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_A, d_A);

    VEC_d *d_B = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_B, d_B);

    // call function
    VEC_d *d_C = (VEC_d*) malloc(sizeof(VEC_d));

    clock_t tic = clock(); 
    matMulVec(d_A, d_B, d_C);
    clock_t toc = clock();
    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    VEC_h *h_C = (VEC_h*) malloc(sizeof(VEC_h));
    VEC_d2h(d_C, h_C);

    verifyMatMulVec(m, n, h_A_val, h_B->h_val, h_C->h_val);

    destroyCSR_d(d_A);
    destroyVEC_d(d_B);
    destroyVEC_d(d_C);
    destroyCSR_h(h_A);
    destroyVEC_h(h_B);
    destroyVEC_h(h_C);
    destroyDN_h(h_A_dn);
}

void testVecMulMat(int m, int n) {
    printf("- TEST matMulVec\n");

    double* h_A_val = (double *) malloc(sizeof(double)*n*m);
    double* h_B_val = (double *) malloc(sizeof(double)*m);

    generateRandomSparseMatrix(m, n, h_A_val, 0.2);
    generateRandomVector(m, h_B_val);
    
    // init csr and vec
    DN_h *h_A_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_A_dn, m, n, h_A_val);

    CSR_h *h_A = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_A_dn, h_A);

    VEC_h *h_B = (VEC_h*) malloc(sizeof(VEC_h));
    initVEC_h(h_B, m, h_B_val);

    // move to gpu
    CSR_d *d_A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_A, d_A);

    VEC_d *d_B = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_B, d_B);

    // call function
    VEC_d *d_C = (VEC_d*) malloc(sizeof(VEC_d));

    clock_t tic = clock(); 
    vecMulMat(d_B, d_A, d_C);
    clock_t toc = clock();
    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    VEC_h *h_C = (VEC_h*) malloc(sizeof(VEC_h));
    VEC_d2h(d_C, h_C);

    verifyVecMulMat(m, n, h_A_val, h_B->h_val, h_C->h_val);

    destroyCSR_d(d_A);
    destroyVEC_d(d_B);
    destroyVEC_d(d_C);
    destroyCSR_h(h_A);
    destroyVEC_h(h_B);
    destroyVEC_h(h_C);
    destroyDN_h(h_A_dn);
}

void testMatAdd(int m, int n) {
    printf("- TEST matAdd\n");

    double* h_A_val = (double *) malloc(sizeof(double)*n*m);
    double* h_B_val = (double *) malloc(sizeof(double)*n*m);

    generateRandomSparseMatrix(m, n, h_A_val, 0.2);
    generateRandomSparseMatrix(m, n, h_B_val, 0.2);
    
    // init csr and vec
    DN_h *h_A_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_A_dn, m, n, h_A_val);

    CSR_h *h_A = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_A_dn, h_A);

    DN_h *h_B_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_B_dn, m, n, h_B_val);

    CSR_h *h_B = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_B_dn, h_B);

    // move to gpu
    CSR_d *d_A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_A, d_A);
    CSR_d *d_B = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_B, d_B);

    // call function   
    CSR_d *d_C = (CSR_d *) malloc(sizeof(CSR_d));

    clock_t tic = clock(); 
    matAdd(d_B, d_A, d_C);
    clock_t toc = clock();
    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    CSR_h *h_C = (CSR_h*) malloc(sizeof(CSR_h));
    CSR_d2h(d_C, h_C);

    DN_h *h_C_dn = (DN_h *) malloc(sizeof(DN_h));
    CSR_h2DN_h(h_C, h_C_dn);

    verifyMatAdd(m, n, h_A_dn->h_val, h_B_dn->h_val, h_C_dn->h_val);

    destroyCSR_d(d_A);
    destroyCSR_d(d_B);
    destroyCSR_d(d_C);
    destroyCSR_h(h_A);
    destroyCSR_h(h_B);
    destroyCSR_h(h_C);
    destroyDN_h(h_A_dn);
    destroyDN_h(h_B_dn);
    destroyDN_h(h_C_dn);
}

void testScalarMulMat(int m, int n, double sc) {
    printf("- TEST matAdd\n");

    double* h_A_val = (double *) malloc(sizeof(double)*n*m);

    generateRandomSparseMatrix(m, n, h_A_val, 0.2);
    
    // init csr 
    DN_h *h_A_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_A_dn, m, n, h_A_val);

    CSR_h *h_A = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_A_dn, h_A);

    // move to gpu
    CSR_d *d_A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_A, d_A);

    // call function   
    CSR_d *d_C = (CSR_d *) malloc(sizeof(CSR_d));

    clock_t tic = clock(); 
    scalarMulMat(sc, d_A, d_C);
    clock_t toc = clock();
    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    CSR_h *h_C = (CSR_h*) malloc(sizeof(CSR_h));
    CSR_d2h(d_C, h_C);

    DN_h *h_C_dn = (DN_h *) malloc(sizeof(DN_h));
    CSR_h2DN_h(h_C, h_C_dn);

    verifyScalarMulMat(m, n, sc, h_A_dn->h_val, h_C_dn->h_val);

    destroyCSR_d(d_A);
    destroyCSR_d(d_C);
    destroyCSR_h(h_A);
    destroyCSR_h(h_C);
    destroyDN_h(h_A_dn);
    destroyDN_h(h_C_dn);
}

void testMatMulMat(int m, int n, int k) {
    printf("- TEST matMulMat\n");

    double* h_A_val = (double *) malloc(sizeof(double)*n*m);
    double* h_B_val = (double *) malloc(sizeof(double)*n*k);

    generateRandomSparseMatrix(m, n, h_A_val, 0.2);
    generateRandomSparseMatrix(n, k, h_B_val, 0.2);
    
    // init csr and vec
    DN_h *h_A_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_A_dn, m, n, h_A_val);

    CSR_h *h_A = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_A_dn, h_A);

    DN_h *h_B_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_B_dn, n, k, h_B_val);

    CSR_h *h_B = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_B_dn, h_B);

    // move to gpu
    CSR_d *d_A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_A, d_A);
    CSR_d *d_B = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_B, d_B);

    // call function   
    CSR_d *d_C = (CSR_d *) malloc(sizeof(CSR_d));

    clock_t tic = clock(); 
    matMulMat(d_A, d_B, d_C);
    clock_t toc = clock();
    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("GPU time %fs, ", time_used);

    CSR_h *h_C = (CSR_h*) malloc(sizeof(CSR_h));
    CSR_d2h(d_C, h_C);

    DN_h *h_C_dn = (DN_h *) malloc(sizeof(DN_h));
    CSR_h2DN_h(h_C, h_C_dn);

    verifyMatMulMat(m, n, k, h_A_dn->h_val, h_B_dn->h_val, h_C_dn->h_val);

    destroyCSR_d(d_A);
    destroyCSR_d(d_B);
    destroyCSR_d(d_C);
    destroyCSR_h(h_A);
    destroyCSR_h(h_B);
    destroyCSR_h(h_C);
    destroyDN_h(h_A_dn);
    destroyDN_h(h_C_dn);
}

void testCopyVec(int n) {
    printf("- TEST copyVec\n");
    double* h_A_val = (double*) malloc( sizeof(double)*n );
    generateRandomVector(n, h_A_val);

    // init vec with value
    VEC_h *h_A = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_A, n, h_A_val);

    // move to gpu vec_d
    VEC_d *d_A = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_A, d_A);

    VEC_h *h_A_copy = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_d *d_A_copy = (VEC_d *) malloc(sizeof(VEC_d));

    copyVEC_h(h_A, h_A_copy);
    copyVEC_d(d_A, d_A_copy);
    
    printf("TEST PASSED\n");

    destroyVEC_d(d_A);
    destroyVEC_d(d_A_copy);
    destroyVEC_h(h_A);
    destroyVEC_h(h_A_copy);
}

void testCopyMat(int m, int n) {
    printf("- TEST copyMat\n");
    double* h_A_val = (double *) malloc(sizeof(double)*n*m);
    generateRandomSparseMatrix(m, n, h_A_val, 0.2);

    DN_h *h_A_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_A_dn, m, n, h_A_val);

    CSR_h *h_A = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_A_dn, h_A);

    CSR_d *d_A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_A, d_A);

    DN_h *h_A_dn_copy = (DN_h *) malloc(sizeof(DN_h));
    CSR_h *h_A_copy = (CSR_h *) malloc(sizeof(CSR_h));
    CSR_d *d_A_copy = (CSR_d *) malloc(sizeof(CSR_d));

    copyCSR_h(h_A, h_A_copy);
    copyCSR_d(d_A, d_A_copy);
    copyDN_h(h_A_dn, h_A_dn_copy);

    printf("TEST PASSED\n");
    
    destroyCSR_d(d_A);
    destroyCSR_d(d_A_copy);
    destroyCSR_h(h_A);
    destroyCSR_h(h_A_copy);
    destroyDN_h(h_A_dn);
    destroyDN_h(h_A_dn_copy);
}


void testVecUseCases() {
    double h_A_val[5] = {1, 2, 3, 4, 5};
    double h_B_val[5] = {2, 3, 4, 5, 6};
    double n = 5;

    // init vec with value
    VEC_h *h_A = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_A, n, h_A_val);

    VEC_h *h_B = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_B, n, h_B_val);

    // move to gpu vec_d
    VEC_d *d_A = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_A, d_A);

    VEC_d *d_B = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_B, d_B);

    printf("A:\n");
    printDVec(n, h_A_val);
    printf("\n");

    printf("B:\n");
    printDVec(n, h_B_val);
    printf("\n");

    // Add
    VEC_d *d_C_add = (VEC_d *) malloc(sizeof(VEC_d)); //no initialization 
    vecAdd(d_A, d_B, d_C_add);
    
    printf("Add:\n");
    printDVec_d(n, d_C_add->d_val);
    printf("\n");

    // Inner Product
    double c_inner = innerProduct(d_A, d_B);
    printf("Inner:\n");
    printf("%f\n", c_inner);

    // Scale
    VEC_d *d_C_sc = (VEC_d *) malloc(sizeof(VEC_d)); // no initialization 
    scalarMulVec(1.5, d_A, d_C_sc);
    printf("* 1.5:\n");
    printDVec_d(n, d_C_sc->d_val);
    printf("\n");

    // Add inplace
    vecAddInPlace(d_A, d_B);
    printf("Inplace Add:\n");
    printDVec_d(n, d_A->d_val);
    printf("\n");

    // Copy
    VEC_h *h_A_copy = (VEC_h *) malloc(sizeof(VEC_h));
    copyVEC_h(h_A, h_A_copy);
    printf("copy A_h:\n");
    printDVec(n, h_A_copy->h_val);
    printf("\n");

    VEC_d *d_A_copy = (VEC_d *) malloc(sizeof(VEC_d));
    copyVEC_d(d_A, d_A_copy);
    printf("copy A_d:\n");
    printDVec_d(n, d_A_copy->d_val);
    printf("\n");
    vecAddInPlace(d_A_copy, d_B);
    printDVec_d(n, d_A_copy->d_val);
    printf("\n");
    printDVec_d(n, d_A->d_val);
    printf("\n");

    destroyVEC_d(d_A);
    destroyVEC_d(d_B);
    destroyVEC_d(d_C_add);
    destroyVEC_d(d_A_copy);
}


void testMatUseCases() {
    double h_A_val[6] = {1, 2, 3, 4, 5, 6};
    double h_B_val[6] = {2, 3, 4, 5, 6, 7};

    double m = 3;
    double n = 2;
    
    // init csr and vec
    DN_h *h_A_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_A_dn, m, n, h_A_val);

    CSR_h *h_A = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_A_dn, h_A);

    DN_h *h_B_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_B_dn, m, n, h_B_val);

    CSR_h *h_B = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_B_dn, h_B);

    CSR_d *d_A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_A, d_A);

    CSR_d *d_B = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_B, d_B);

    printf("A:\n");
    printMatrix_d(m, n, d_A->d_val);
    printf("\n");

    printf("B:\n");
    printMatrix_d(m, n, d_B->d_val);
    printf("\n");

    printf("Add:\n");
    CSR_d *d_C_add = (CSR_d *) malloc(sizeof(CSR_d));
    matAdd(d_A, d_B, d_C_add);
    printMatrix_d(d_C_add->m, d_C_add->n, d_C_add->d_val);
    printf("\n");

    printf("A * 1.5:\n");
    CSR_d *d_C_sc = (CSR_d *) malloc(sizeof(CSR_d));
    scalarMulMat(1.5, d_A, d_C_sc);
    printMatrix_d(d_C_add->m, d_C_add->n, d_C_sc->d_val);
    printf("\n");

    double h_D_val[2] = {2, 3};
    VEC_h *h_D = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_D, n, h_D_val);
    VEC_d *d_D = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_D, d_D);
    printf("Vec D:\n");
    printDVec_d(d_D->n, d_D->d_val);
    printf("\n");
    VEC_d *d_C_mat_vec = (VEC_d *) malloc(sizeof(VEC_d));
    matMulVec(d_A, d_D, d_C_mat_vec);
    printf("Vec A * D:\n");
    printDVec_d(d_C_mat_vec->n, d_C_mat_vec->d_val);
    printf("\n");

    double h_E_val[3] = {2, 3, 4};
    VEC_h *h_E = (VEC_h *) malloc(sizeof(VEC_h));
    initVEC_h(h_E, m, h_E_val);
    VEC_d *d_E = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_h2d(h_E, d_E);
    printf("Vec E:\n");
    printDVec_d(d_E->n, d_E->d_val);
    printf("\n");
    VEC_d *d_C_vec_mat = (VEC_d *) malloc(sizeof(VEC_d));
    vecMulMat(d_E, d_A, d_C_vec_mat);
    printf("Vec A * E:\n");
    printDVec_d(d_C_vec_mat->n, d_C_vec_mat->d_val);
    printf("\n");

    double h_F_val[8] = {2, 3, 4, 5, 6, 7, 8, 9};
    double k = 4;
    DN_h *h_F_dn = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(h_F_dn, n, k, h_F_val);
    CSR_h *h_F = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(h_F_dn, h_F);
    CSR_d *d_F = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(h_F, d_F);

    printf("Mat F:\n");
    printMatrix_d(d_F->m, d_F->n, d_F->d_val);
    CSR_d *d_C_mat_mat = (CSR_d *) malloc(sizeof(CSR_d));
    matMulMat(d_A, d_F, d_C_mat_mat);
    printf("A * F\n");
    printMatrix_d(d_C_mat_mat->m, d_C_mat_mat->n, d_C_mat_mat->d_val);


}


int main(int argc, char**argv) {
    checkCublasErrors(cublasCreate(&cublasHandle));
    checkCusparseErrors(cusparseCreate(&cusparseHandle));

    testVecAdd(1000000);
    testVecAddInPlace(1000000);
    testInnerProduct(1000000);
    testScalarMulVec(1000000, 3.14159);
    testMatMulVec(1000, 500);
    testVecMulMat(1000, 500);
    testMatAdd(1000, 500);
    testMatMulMat(1000, 500, 100);
    testScalarMulMat(3.14159, 1000, 500);
    testCopyVec(1000000);
    testCopyMat(1000, 500);

    // testVecUseCases();
    // testMatUseCases();

    cublasDestroy(cublasHandle);
    cusparseDestroy(cusparseHandle);
 
    return 0;
}
 