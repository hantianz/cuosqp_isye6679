//
// Created by Hantian Zhang on 4/17/2022.
//

//
// Created by Hantian Zhang on 4/4/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INF (1 << 15)
#define THREADS_PER_BLOCK   (1024)

# ifndef c_max
#  define c_max(a, b) (((a) > (b)) ? (a) : (b))
# endif /* ifndef c_max */

# ifndef c_min
#  define c_min(a, b) (((a) < (b)) ? (a) : (b))
# endif /* ifndef c_min */

/* CSR matrix */
typedef struct {
    int m; // number of rows
    int n; //num of columns
    int nnz; // num of nonzero entries
    double* h_val; // Points to the data array of length nnz that holds all nonzero values of A in row-major format.
    int* h_rowPtr; // Points to the integer array of length m+1 that holds indices into the arrays csrColIndA and csrValA
    int* h_colInd; // Points to the integer array of length nnz that contains the column indices of the corresponding elements in array csrValA
} CSR_h;

void initCSR_h(CSR_h *mat, int m, int n, int nnz, double *h_val, int *h_rowPtr, int *h_colInd) {
    mat->m = m;
    mat->n = n;
    mat->nnz = nnz;
    mat->h_val = h_val;
    mat->h_rowPtr = h_rowPtr;
    mat->h_colInd = h_colInd;
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


/* Dense vector*/
typedef struct {
    int n; // length of vector
    double* h_val;
} VEC_h;



void initVEC_h(VEC_h *v, int n, double *h_val) {
    v->n = n;
    v->h_val = h_val;
}


void copyVEC_h(VEC_h *vec_src, VEC_h *vec_dest) {
    double *h_val = (double*) malloc(sizeof(double)*vec_src->n);
    memcpy(h_val, vec_src->h_val, sizeof(double)*vec_src->n);
    initVEC_h(vec_dest, vec_src->n, h_val);
}


/* Dense matrix*/
typedef struct {
    int m;
    int n;
    double* h_val;
} DN_h;


void initDN_h(DN_h *mat, int m, int n, double *h_val) {
    mat->m = m;
    mat->n = n;
    mat->h_val = h_val;
}


void copyDN_h(DN_h *mat_src, DN_h *mat_dest) {
    double *h_val = (double*) malloc(sizeof(double)*(mat_src->n * mat_src->m));
    memcpy(h_val, mat_src->h_val, sizeof(double)*(mat_src->n * mat_src->m));
    initDN_h(mat_dest, mat_src->m, mat_src->n, h_val);
}

/* ------------------------------- destructor -------------------------------*/
void destroyVEC_h(VEC_h *v) {
    free(v->h_val);
    free(v);
}

void destroyDN_h(DN_h *mat) {
    free(mat->h_val);
    free(mat);
}


void destroyCSR_h(CSR_h *mat) {
    free(mat->h_val);
    free(mat->h_colInd);
    free(mat->h_rowPtr);
    free(mat);
}


/* ------------------------------- type conversion -------------------------------*/



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
        h_colInd = (int *) malloc(sizeof(int) *nnz);
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


void vecAdd(VEC_h *A, VEC_h *B, VEC_h *C){
    C->n = A->n;
    C->h_val = (double *) malloc(sizeof(double) *(A->n));
    for (int i = 0; i < A->n; ++i)
        C->h_val[i] = A->h_val[i] + B->h_val[i];
}

void vecAddInPlace(VEC_h *A, VEC_h *B){
    for (int i = 0; i < A->n; ++i)
        A->h_val[i] = A->h_val[i] + B->h_val[i];
}

double innerProduct(VEC_h *A, VEC_h *B){
    double product = 0.0;
    for (int i = 0; i < A->n; ++i)
        product += A->h_val[i] * B->h_val[i];
    return product;
}

void scalarMulVec(double val, VEC_h *A, VEC_h *B)
{
    B->n = A->n;
    B->h_val = (double *) malloc(sizeof(double) *(A->n));
    for (int i = 0; i < A->n; ++i)
        B->h_val[i] = A->h_val[i] * val;
}

void matMulMat(DN_h *A, DN_h *B, DN_h *C){
    C->h_val = (double *) malloc(sizeof(double) *(A->m * B->n));
    C->m = A->m;
    C->n = B->n;
    for (int i = 0; i < A->m; ++i)
        for (int j = 0; j < C->n; ++j) {
            C->h_val[i*C->n+j] = 0;
            for (int k = 0; k < A->n; ++k)
                C->h_val[i*C->n+j] += A->h_val[i*A->n+k] * B->h_val[k*B->n+j];
        }
}

// calculate vector A* matrix B and store it in vector C
void vecMulMat(VEC_h *A, DN_h *B, VEC_h *C) {
    C->h_val = (double *) malloc(sizeof(double) *(B->n));
    C->n = B->n;
    for (int i = 0; i < B->n; ++i) {
        C->h_val[i] = 0;
        for (int j = 0; j < A->n; ++j)
            C->h_val[i] += A->h_val[j] * B->h_val[j*B->n+i];
    }
}


// calculate matrix A* vector B and store it in vector C
void matMulVec(DN_h *A, VEC_h *B, VEC_h *C){
    C->h_val = (double *) malloc(sizeof(double) *(A->m));
    C->n = A->m;
    for (int i = 0; i < A->m; ++i)
    {
        C->h_val[i] = 0;
        for (int j = 0; j < B->n; ++j)
            C->h_val[i] += A->h_val[i*A->n + j] * B->h_val[j];
    }

}

// calculate sum of two matrices
void matAdd(DN_h *A, DN_h *B, DN_h *C){
    C->h_val = (double *) malloc(sizeof(double) *(A->m * A->n));
    C->n = A->n;
    C->m = A->m;
    for (int i = 0; i < A->m; ++i)
        for (int j = 0; j < A->n; ++j)
            C->h_val[i*A->n+j] = A->h_val[i*A->n+j] + B->h_val[i*A->n+j];
}

void scalarMulMat(double val, DN_h *A, DN_h *B)
{
    B->h_val = (double *) malloc(sizeof(double) *(A->m * A->n));
    B->n = A->n;
    B->m = A->m;
    for (int i = 0; i < A->m * A->n; ++i)
        B->h_val[i] = A->h_val[i] * val;
}



//// calculate transpose of a matrix A and store it in matrix B
void transpose(DN_h *A, DN_h *B){
    B->h_val = (double *) malloc(sizeof(double) *(A->m * A->n));
    B->n = A->m;
    B->m = A->n;
    for (int i = 0; i < A->m; ++i)
        for (int j = 0; j < A->n; ++j)
            B[j*A->m+i] = A[i*A->n+j];
}



void inverseDiag(DN_h *A, DN_h *B){
    copyDN_h(A,B);
    for (int i =0; i < A->m; ++i)
    {
        B->h_val[i*A->n+i] = 1 / A->h_val[i*A->n+i];
    }
}



// fill a diagonal n*n matrix A with val on each diagonal entry
void initializeDiagMat(int n, double val, DN_h *A){
    A->n = n;
    A->m = n;
    A->h_val = (double *) malloc(sizeof(double) *(n*n));
    for (int i =0; i < n; ++i)
    {
        A->h_val[i*n+i] = val;
    }
}

//// min(l) max(u) projection of an vector A to B
void vecMinMaxProj(VEC_h *A, VEC_h *l, VEC_h *u, VEC_h *B){
    B->n = A->n;
    B->h_val = (double *) malloc(sizeof(double) *(A->n));
    for (int i = 0; i < A->n; ++i)
        if (A->h_val[i] < l->h_val[i])
            B->h_val[i] = l->h_val[i];
        else if (A->h_val[i] > u->h_val[i])
            B->h_val[i] = u->h_val[i];
        else
            B->h_val[i] = A->h_val[i];
}



double normInf(VEC_h *A){
    double maxValue = -INF;
    for (int i = 0; i < A->n; ++i)
        if (fabs(A->h_val[i]) > maxValue)
            maxValue = fabs(A->h_val[i]);
    return maxValue;
}



// if l==u, penalty = 1000*rho, otherwise penalty = rho
void calculateR(int n, DN_h *R, VEC_h *l, VEC_h *u, double rho){
    R->m = n;
    R->n = n;
    R->h_val = (double *) malloc(sizeof(double) *(n * n));
    for (int i = 0; i < n; ++i)
    {
        if (l->h_val[i] == l->h_val[i])
        {
            R->h_val[i*n+i] = rho * 1000;
        }
        else
        {
            R->h_val[i*n+i] = rho;
        }
    }
}
void getDiagonal(DN_h *K, DN_h *M){
    M->n = K->n;
    M->m = K->m;
    M->h_val = (double *) malloc(sizeof(double) *(K->n * K->m));
    int idx;
    if (K->n < K->m)
        idx = K->n;
    else
        idx = K->m;
    for (int i = 0; i < idx; i++)
        M->h_val[i*K->n+i] = K->h_val[i*K->n+i];
}


// calculate objective value obj = 1/2 * x^T * P * x + q^T * X
double objValue(DN_h *P, VEC_h *q, VEC_h *x)
{
    VEC_h *temp = (VEC_h *) malloc(sizeof(VEC_h));
    vecMulMat(x, P, temp);
    double obj = 0.5 * innerProduct(temp, x) + innerProduct(q, x);
    //destroyVEC_h(temp);
    return obj;
}

// calculate if the program should terminate
int termination(VEC_h *x, VEC_h *y, VEC_h *z, DN_h *P, VEC_h *Q, DN_h *A, DN_h *AT, double epsilonPrimal, double epsilonDual){
//    clock_t tic = clock();
    // temp1 = A*x
    VEC_h *temp1 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp2 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *residualPrimal = (VEC_h *) malloc(sizeof(VEC_h));

    //calculate residual of Primal
    matMulVec(A, x, temp1); // temp1 = A*x;

    scalarMulVec(-1, z, temp2); // temp2 = -z;
    vecAdd(temp1, temp2, residualPrimal); // residualPrimal = A*x - z;

    //calculate residual of Dual
    VEC_h *temp3 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp4 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp5 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *residualDual = (VEC_h *) malloc(sizeof(VEC_h));

    matMulVec(P, x, temp3); // temp3 = P*x;
    matMulVec(AT, y, temp4); // temp4 = A^T * y;
    vecAdd(temp3, temp4, temp5); // temp5 = temp3 + temp4;
    vecAdd(temp5, Q, residualDual); // residualDual = temp5 + Q;
    double residualPrimalDouble = normInf(residualPrimal);
    double residualDualDouble = normInf(residualDual);
//    destroyVEC_h(temp1);
//    destroyVEC_h(temp2);
//    destroyVEC_h(temp3);
//    destroyVEC_h(temp4);
//    destroyVEC_h(temp5);
//    destroyVEC_h(residualPrimal);
//    destroyVEC_h(residualDual);
    printf("primalResidual:%.3f, dualResidual:%.3f\n", residualPrimalDouble, residualDualDouble);
    //if (residualPrimalDouble <= epsilonPrimal)
//    clock_t toc = clock();
//    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
//    printf("Termination time %fs, \n", time_used);
    if ((residualPrimalDouble <= epsilonPrimal) && (residualDualDouble <= epsilonDual))
        return 1;
    else
        return 0;
}


// algorithm 3 with GPU
void solveKKT(int n, int m, VEC_h *x, VEC_h *y, VEC_h *z, DN_h *P, VEC_h *Q, DN_h *A, DN_h *AT, VEC_h *l, VEC_h *u, DN_h *R, VEC_h *xNext, VEC_h *zNext, double rho, double sigma, double epsilon)
{
//    clock_t tic = clock();

    DN_h *K = (DN_h *) malloc(sizeof(DN_h));
    DN_h *I = (DN_h *) malloc(sizeof(DN_h));
    initializeDiagMat(n, 1, I);
    DN_h *temp1 = (DN_h *) malloc(sizeof(DN_h));
    DN_h *temp2 = (DN_h *) malloc(sizeof(DN_h));
    DN_h *temp3 = (DN_h *) malloc(sizeof(DN_h));
    DN_h *temp4 = (DN_h *) malloc(sizeof(DN_h));
    matMulMat(AT, R, temp1); // temp1 = A^T * R;
    matMulMat(temp1, A, temp2); // temp2 = A^T * R * A;
    scalarMulMat(sigma, I, temp3);
    matAdd(temp2, temp3, temp4);
    matAdd(temp4, P, K);




    DN_h *M = (DN_h *) malloc(sizeof(DN_h));
    DN_h *MINV = (DN_h *) malloc(sizeof(DN_h));

//    clock_t toc = clock();
//    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
//    printf("KKT Init time %fs, \n", time_used);
//    tic = clock();
    //CSR_d2h(K, K_h);
    //getDiagonal(K_h, M_h);
    getDiagonal(K, M);
//    printCSRd(K);
//    printCSRd(M);
    // printf("%d, %d, %d, %d, %d", MINV->m, MINV->n, r->n, MINV->nnz, M->nnz);
    // printDVec_d(M->nnz, M->d_val);
    // printDVec_d(MINV->nnz, MINV->d_val);

    //CSR_h *MINV_h = (CSR_h *) malloc(sizeof(CSR_h));
    //CSR_h2d(M_h, M);
    //inverseDiag(M_h,MINV_h);
    //CSR_h2d(MINV_h, MINV);
    inverseDiag(M, MINV);
//    printCSRd(MINV);
//    exit(-1);
 //   toc = clock();
 //   time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
 //   printf("KKT Diag time %fs, \n", time_used);
 //   tic = clock();
    VEC_h *b = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp5 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp6 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp7 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp8 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp9 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp10 = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *temp11 = (VEC_h *) malloc(sizeof(VEC_h));
    matMulVec(R, z, temp5);
    scalarMulVec(-1, y, temp6);
    vecAdd(temp5, temp6, temp7); // temp7 = Rz - y;
    matMulVec(AT, temp7, temp8); // temp8 = A^T(Rz - y);
    scalarMulVec(sigma, x, temp9); // temp9 = sigma * x;
    scalarMulVec(-1, Q, temp10); // temp10 = -q;
    vecAdd(temp9, temp10, temp11); // temp11 = sigma * x - q;
    vecAdd(temp11, temp8, b); // b = sigma * x - q + A^T(Rz - y);

    //initialize r0=Kx-b, y0, p0
    VEC_h *r = (VEC_h *) malloc(sizeof(VEC_h));
    matMulVec(K, x, temp8); // temp8 = Kx;
    scalarMulVec(-1, b, temp9); // temp9 = -b;

    // printCSRd(R);
    // printVecd(z);
    // printVecd(temp5);
    // printVecd(temp6);
    // printVecd(temp8);
    // printVecd(temp9);
    // printVecd(temp11);
    // printVecd(b);
    // exit(0);

    vecAdd(temp8, temp9, r);

    //intitialize y0 = M^(-1)*r;
    VEC_h *y_kkt = (VEC_h *) malloc(sizeof(VEC_h));

    matMulVec(MINV, r, y_kkt); // y0 = M^(-1)*r

    // intitialize p0 = -y0;
    VEC_h *p = (VEC_h *) malloc(sizeof(VEC_h));
    scalarMulVec(-1, y_kkt, p);

    // printf("r:");
    // printVecd(r);

    int k = 0;
    double normR = normInf(r);
    double normB = normInf(b);

    // printf("normR :%f\n", normR);
    // printf("normB :%f\n", normB);

 //   toc = clock();
 //   time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
  //  printf("KKT Before Loop time %fs, \n", time_used);
    while (normR > epsilon * normB)
    {
        //tic = clock();
        //calculate a^k
        double alpha;
        double temp12, temp13;
        temp12 = innerProduct(r, y_kkt); // calculate Numerator r^T * y;
        vecMulMat(p, K, temp8); // temp8 = p^T * K
        temp13 = innerProduct(temp8, p); // calculate denominator p^T * K * p;
        alpha = temp12 / temp13;
        //calculate x^k+1
        scalarMulVec(alpha, p, temp8);

        if (k==0) {
            vecAdd(x, temp8, xNext); // x = x + alpha * p;
        }
        else{
            vecAddInPlace(xNext, temp8);
        }
        //calculate r^k+1
        matMulVec(K, p, temp8); // temp8 = K * p;
        scalarMulVec(alpha, temp8, temp9); // temp9 = alpha * K * p;
        vecAddInPlace(r, temp9);

        //calculate y^k+1
        matMulVec(MINV, r, y_kkt); // y = M^(-1) * rNew;
        //calculate beta^k+1
        double beta;
        temp13 = innerProduct(r, y_kkt); // calculate Numerator rNew^T * yNew;
        beta = temp13 / temp12; // beta = ( rNew^T * yNew) / ( r^T * y)
        //calculate p^k+1
        scalarMulVec(-1, y_kkt, temp8); // temp8 = -y
        scalarMulVec(beta, p, temp9); // temp9 = beta * p;
        vecAdd(temp8, temp9, p);
        k+=1;

        normR = normInf(r);
  //      toc = clock();
   //     time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
        //printf("%d, KKT Loop time %fs, \n", k, time_used);
        //printf("normR :%f\n", normR);
        //printf("normB :%f\n", normB);
        if (k>500)
            break;
    }
    matMulVec(A, xNext, zNext);

    // printVecd(xNext);
    // exit(0);

//    destroyVEC_h(temp5);
//    destroyVEC_h(temp6);
//    destroyVEC_h(temp7);
//    destroyVEC_h(temp8);
//    destroyVEC_h(temp9);
//    destroyVEC_h(temp10);
//    destroyVEC_h(temp11);
//    destroyVEC_h(b);
//    destroyVEC_h(r);
//    destroyVEC_h(y_kkt);
//    destroyVEC_h(p);
//
//    destroyDN_h(temp1);
//    destroyDN_h(temp2);
//    destroyDN_h(temp3);
//    destroyDN_h(temp4);
//    destroyDN_h(K);
//    destroyDN_h(I);
//    destroyDN_h(M);
//    destroyDN_h(MINV);

    //destroyCSR_h(M_h);
    //destroyCSR_h(MINV_h);

    //destroyCSR_h(K_h);
    //destroyCSR_h(I_h);

}

//void twoD2oneD(int n, int m, double* A, double *B)
//{
//    for (int i = 0; i < n; i++)
//        for (int j = 0; j < m; j++)
//            B[i*n+j] = *((A+i*n) + j);
//}

//
void readSparseMatrix(char * FILENAME, CSR_h *DST, int *n, int *m)
{
    FILE *myfile;
    int nnz;
    myfile = fopen(FILENAME, "r");
    fscanf(myfile,"%i",n);
    fscanf(myfile,"%i",m);
    fscanf(myfile,"%i",&nnz);

    //printf("%d, %d, %d", *n, *m, nnz);
    double *h_val = (double *) malloc(sizeof(double) * nnz);
    int *h_rowPtr = (int *) malloc(sizeof(int) * (*n+1));
    int *h_colInd = (int *) malloc(sizeof(int) * nnz);
    for(int i = 0; i < nnz; i++)
    {
        fscanf(myfile,"%lf",h_val+i);
    }
//    for(int i = 0; i < nnz; i++)
//    {
//        printf("%lf,",h_val[i]);
//    }
//    printf("\n");
    for(int i = 0; i < *n + 1; i++)
    {
        fscanf(myfile,"%i",h_rowPtr+i);
    }
//    for(int i = 0; i < *n + 1; i++)
//    {
//        printf("%i,",h_rowPtr[i]);
//    }
//    printf("\n");
    for(int i = 0; i < nnz; i++)
    {
        fscanf(myfile,"%i",h_colInd+i);
    }
    initCSR_h(DST, *n, *m, nnz, h_val, h_rowPtr, h_colInd);
}

void readSparseMatrix_nom(char * FILENAME, CSR_h *DST)
{
    FILE *myfile;
    int n, m, nnz;
    myfile = fopen(FILENAME, "r");
    fscanf(myfile,"%i",&n);
    fscanf(myfile,"%i",&m);
    fscanf(myfile,"%i",&nnz);

    //printf("%d, %d, %d\n", n, m, nnz);
    double *h_val = (double *) malloc(sizeof(double) * nnz);
    int *h_rowPtr = (int *) malloc(sizeof(int) * (n+1));
    int *h_colInd = (int *) malloc(sizeof(int) * nnz);
    for(int i = 0; i < nnz; i++)
    {
        fscanf(myfile,"%lf",h_val+i);
    }

    for(int i = 0; i < n + 1; i++)
    {
        fscanf(myfile,"%i",h_rowPtr+i);
    }
    for(int i = 0; i < nnz; i++)
    {
        fscanf(myfile,"%i",h_colInd+i);
    }
    initCSR_h(DST, n, m, nnz, h_val, h_rowPtr, h_colInd);

}

void readVector(char * FILENAME, VEC_h *DST)
{
    FILE *myfile;
    int n;
    myfile = fopen(FILENAME, "r");
    fscanf(myfile,"%i",&n);
    double *h_val = (double *) malloc(sizeof(double) * n);
    for(int i = 0; i < n; i++)
    {
        fscanf(myfile,"%lf",h_val+i);
    }
    initVEC_h(DST, n, h_val);
}

int main() {
    clock_t tic = clock();
    int n,m;
    double sigma = 0.000001, alpha=1.6, rho =  9;//9.054556534215896;
    char testcase[40]="data/instance-3x4/";
    //get n and m
    // input
//    //case 1
//    n = 2;
//    m = 2;
//  case 2
//    n = 10;
//    m = 10;
//    double P_h[n*n],Q_h_val[n], A_h[m*n], AT_h[n*m], l_h_val[m],u_h_val[m];
    CSR_h *A_csrh = (CSR_h *) malloc(sizeof(CSR_h));
//    CSR_h *AT_csrh = (CSR_h *) malloc(sizeof(CSR_h));
    CSR_h *P_csrh = (CSR_h *) malloc(sizeof(CSR_h));
//    DN_h *AT_dh_h = (DN_h *) malloc(sizeof(DN_h));
//    DN_h *A_dh_h = (DN_h *) malloc(sizeof(DN_h));
    VEC_h *l = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *u = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *Q = (VEC_h *) malloc(sizeof(VEC_h));
    char testDST[40];
    strcpy(testDST, testcase);
    char filename[40] = "A.txt";
    strcat(testDST, filename);
    //printf("%s\n", testDST);
    readSparseMatrix(testDST,A_csrh,&m,&n);
//    CSR_h2DN_h(A_csrh, A_dh_h);
//    copyDN_h(A_dh_h, AT_dh_h);
//    AT_dh_h->n = A_dh_h->m;
//    AT_dh_h->m = A_dh_h->n;
//    transpose(m, n, A_dh_h->h_val, AT_dh_h->h_val);
//    DN_h2CSR_h(AT_dh_h, AT_csrh);

    strcpy(testDST, testcase);
    char filename_P[40] = "P.txt";
    strcat(testDST, filename_P);
    readSparseMatrix_nom(testDST,P_csrh);

    strcpy(testDST, testcase);
    char filename_Q[40] = "q.txt";
    strcat(testDST, filename_Q);
    readVector(testDST,Q);

    strcpy(testDST, testcase);
    char filename_l[40] = "l.txt";
    strcat(testDST, filename_l);
    readVector(testDST,l);


    strcpy(testDST, testcase);
    char filename_u[40] = "u.txt";
    strcat(testDST, filename_u);
    readVector(testDST,u);
    //readSparseMatrix("")
    // case 1
//    P_h[0] = 0.01;
//    P_h[1] = 0.0;
//    P_h[2] = 0.0;
//    P_h[3] = 0.2889654;
//    Q_h_val[0] = -1.07296862;
//    Q_h_val[1] = 0.86540763;
//    A_h[0] = 0.0;
//    A_h[1] = 0.0;
//    A_h[2] = 0.0;
//    A_h[3] = -2.3015387;
//    AT_h[0] = 0.0;
//    AT_h[1] = 0.0;
//    AT_h[2] = 0.0;
//    AT_h[3] = -2.3015387;
//    l_h_val[0] = 0 - INF;
//    l_h_val[1] = 0 - INF;
//    u_h_val[0] = 0.22957721;
//    u_h_val[1] = -2.11756839;
    //   get_input(n,m,P,Q,l,u,A);
    //get AT = A^T
    //transpose(m,n, A_h, AT_h);
    //initialize x,y,z
    //double R_h[m];
    //double A_2d[m*n];
    //double AT_2d[n*m];
    //double P_2d[n*n];

    //DN_h *A_dh_h = (DN_h *) malloc(sizeof(DN_h));
    //twoD2oneD(m, n, A_h, A_2d);
    //initDN_h(A_dh_h, m, n, A_h);
    //DN_h2CSR_h(A_dh_h, A_csrh);
    DN_h *A = (DN_h *) malloc(sizeof(DN_h));
    DN_h *P = (DN_h *) malloc(sizeof(DN_h));

    CSR_h2DN_h(A_csrh, A);

    //DN_h *AT_dh_h = (DN_h *) malloc(sizeof(DN_h));
    //twoD2oneD(n, m, AT_h, AT_2d);
    //initDN_h(AT_dh_h, n, m, AT_h);
    //DN_h2CSR_h(AT_dh_h, AT_csrh);
    DN_h *AT = (DN_h *) malloc(sizeof(DN_h));

    //CSR_h2d(AT_csrh, AT);
    //tic = clock();
    transpose(A,AT);
    //toc = clock();
    //time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    //printf("Transpose time %fs, \n", time_used);
    //printf("%ld\n", sizeof(DN_h));
    //DN_h *P_dh_h = (DN_h *) malloc(sizeof(DN_h));
    //initDN_h(P_dh_h, n, n, P_h);
    //DN_h2CSR_h(P_dh_h, P_csrh);
    //DN_h *P = (DN_h *) malloc(sizeof(DN_h));
    CSR_h2DN_h(P_csrh, P);
    //tic = clock();
    //DN_h *R_dh_h = (DN_h *) malloc(sizeof(DN_h));
    //initDN_h(R_dh_h, m, m, R_h);
    DN_h *R = (DN_h *) malloc(sizeof(DN_h));
    calculateR(m, R, l, u, rho);
    DN_h *RINV = (DN_h *) malloc(sizeof(DN_h));
    //CSR_h *RINV_h = (CSR_h *) malloc(sizeof(CSR_h));
    inverseDiag(R, RINV);
    //CSR_h2d(RINV_h, RINV);
    //toc = clock();
    //time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    //printf("R time %fs, \n", time_used);
    double x_h_val[n];
    double y_h_val[m];
    double z_h_val[m];
    memset(x_h_val, 0, sizeof(double)*n);
    memset(y_h_val, 0, sizeof(double)*m);
    memset(z_h_val, 0, sizeof(double)*m);

    VEC_h *x = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *y = (VEC_h *) malloc(sizeof(VEC_h));
    VEC_h *z = (VEC_h *) malloc(sizeof(VEC_h));


    initVEC_h(x, n, x_h_val);
    initVEC_h(y, m, y_h_val);
    initVEC_h(z, m, z_h_val);
//
//    VEC_d *x = (VEC_d *) malloc(sizeof(VEC_d));
//    VEC_d *y = (VEC_d *) malloc(sizeof(VEC_d));
//    VEC_d *z = (VEC_d *) malloc(sizeof(VEC_d));
//    VEC_d *l = (VEC_d *) malloc(sizeof(VEC_d));
//    VEC_d *u = (VEC_d *) malloc(sizeof(VEC_d));
//    VEC_d *Q = (VEC_d *) malloc(sizeof(VEC_d));
//
//    VEC_h2d(x_h, x);
//    VEC_h2d(y_h, y);
//    VEC_h2d(z_h, z);
//    VEC_h2d(l_h, l);
//    VEC_h2d(u_h, u);
//    VEC_h2d(Q_h, Q);

    // initialize sigma and alpha
    // initialize epsilon, now we assume epsilon is a constant
    // todo add a function to calculate epsilon in each iteration;
    double epsilon = 0.00001;
    double epsilonPrimal = 0.0001;
    double epsilonDual = 0.1;
    //eps = calc_eps();
    int k = 0;

    while (!termination(x, y, z, P, Q, A, AT, epsilonPrimal, epsilonDual) && k <= 100)
    {

        VEC_h *xNext = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *zNext = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *xOld = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *yOld = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *zOld = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *xDiff = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *yDiff = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *zDiff = (VEC_h *) malloc(sizeof(VEC_h));
        copyVEC_h(x, xOld);
        copyVEC_h(y, yOld);
        copyVEC_h(z, zOld);
        copyVEC_h(x, xNext);
        copyVEC_h(z, zNext);

        //calculate the penalty matrix R and its inverse RINV
        //tic = clock();
        solveKKT(n, m, x, y, z, P, Q, A, AT, l,u, R, xNext, zNext, rho, sigma, epsilon);
        //toc = clock();
        //time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
        //printf("KKT time %fs, \n", time_used);
        // update x
        //tic = clock();
        VEC_h *temp1 = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *temp2 = (VEC_h *) malloc(sizeof(VEC_h));
//        printf("%d\n", k);
//        printVecd(xNext);
//        printVecd(zNext);
        scalarMulVec(alpha, xNext, temp1); // temp1 = alpha * xNext;
        scalarMulVec(1 - alpha, x, temp2); // temp2 = (1 - alpha) * x;
        vecAdd(temp1, temp2, x); // x = alpha * xNext +  (1 - alpha) * x;

        //calculate xDiff
        scalarMulVec(-1, xOld, temp1); // temp1 = alpha * xNext;
        vecAdd(x, temp1, xDiff);

        // update z
        VEC_h *temp3 = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *temp4 = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *temp5 = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *temp6 = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *temp7 = (VEC_h *) malloc(sizeof(VEC_h));
        VEC_h *zNextReal = (VEC_h *) malloc(sizeof(VEC_h));

        scalarMulVec(alpha, zNext, temp3); // temp3 = alpha * zNext;
        scalarMulVec(1 - alpha, z, temp4); // temp4 = (1 - alpha) * z;

        matMulVec(RINV,y, temp5 ); // temp5 = R^(-1) * y;


        vecAdd(temp3, temp4, temp6);  // temp6 = alpha * zNext + (1 - alpha) * z;
        vecAdd(temp5, temp6, temp7);    // temp7 = alpha * zNext + (1 - alpha) * z + R^(-1) * y

        // printVecd(xNext);
        // printVecd(zNext);
        // printVecd(temp1);
        // printVecd(temp2);
        // printVecd(temp3);
        // printVecd(temp4);
        // printVecd(temp5);
        // printVecd(temp6);
        // printVecd(temp7);
        // exit(0);

        vecMinMaxProj(temp7, l, u, zNextReal); // zNextReal is the projection of temp7

        //calculate zDiff
        scalarMulVec(-1, zOld, temp1); // temp1 = alpha * xNext;
        vecAdd(zNextReal, temp1, zDiff);

        // update y
        scalarMulVec(-1, zNextReal, temp3); // temp3 = -zNextReal;
        vecAdd(temp6, temp3, temp4); // temp4 = alpha * zNext + (1 - alpha) * z - zNextReal;
        matMulVec(R, temp4, temp5); // temp5 = R * (alpha * zNext + (1 - alpha) * z - zNextReal);
        vecAddInPlace(y, temp5); // y = y + temp5;

        //calculate yDiff
        scalarMulVec(-1, yOld, temp1); // temp1 = alpha * xNext;
        vecAdd(y, temp1, yDiff);

        // update z^k to z^(k+1)


        // printf("%d\n", zNextReal->n);
        // printVecd(zNextReal);
        copyVEC_h(zNextReal, z);

        printf("\n");
        printf("Round:%d, obj:%.6f\n", k, objValue(P,Q,x));
        printf("norm diff x:%.6f, norm diff y:%.6f, norm diff z:%.6f\n", normInf(xDiff),normInf(yDiff),normInf(zDiff));
        //printVecd(x);
        k += 1;

//        destroyVEC_h(temp1);
//        destroyVEC_h(temp2);
//        destroyVEC_h(temp3);
//        destroyVEC_h(temp4);
//        destroyVEC_h(temp5);
//        destroyVEC_h(temp6);
//        destroyVEC_h(temp7);
//        destroyVEC_h(xNext);
//        destroyVEC_h(zNext);
//        destroyVEC_h(xOld);
//        destroyVEC_h(yOld);
//        destroyVEC_h(zOld);
//        destroyVEC_h(xDiff);
//        destroyVEC_h(yDiff);
//        destroyVEC_h(zDiff);
    }

//    destroyVEC_h(x);
//    destroyVEC_h(y);
//    destroyVEC_h(z);
//    destroyVEC_h(l);
//    destroyVEC_h(u);
//    destroyVEC_h(Q);
//    destroyDN_h(A);
//    destroyDN_h(AT);
//    destroyDN_h(R);
//    destroyDN_h(RINV);
    clock_t toc = clock();
    double time_used = (double) (toc - tic) / CLOCKS_PER_SEC;
    printf("Total time %fs, \n", time_used);
    return 0;
}
