//
// Created by Hantian Zhang on 4/4/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cusparse_cublas_funcs.cu"

#define INF (1 << 28)
#define THREADS_PER_BLOCK   (1024)

# ifndef c_max
#  define c_max(a, b) (((a) > (b)) ? (a) : (b))
# endif /* ifndef c_max */

# ifndef c_min
#  define c_min(a, b) (((a) < (b)) ? (a) : (b))
# endif /* ifndef c_min */




__global__ void vec_bound_kernel(int n,
                              VEC_d *A,
                              VEC_d *l,
                              VEC_d *u,
                              VEC_d *B
){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n) {
        B->d_val[i] = c_min(u->d_val[i], c_max(l->d_val[i],A->d_val[i]));
    }
}


void vecMinMaxProj(int n,
                   VEC_d *A,
                   VEC_d *l,
                   VEC_d *u,
                   VEC_d *B) {

    int number_of_blocks = (n / THREADS_PER_BLOCK) + 1;

    vec_bound_kernel<<<number_of_blocks, THREADS_PER_BLOCK>>>(n, A, l, u, B);
}

//// calculate transpose of a matrix A and store it in matrix B
//void transpose(int n, int m, double A[][m], double B[][n]){
//    for (int i = 0; i < n; ++i)
//        for (int j = 0; j < m; ++j)
//            *((B+j*n) + i) = *((A+i*n) + j);
//}


void get_input(int n, int m, double **P, double *Q, double *l, double *u, double **A){
    FILE *myfile;
    int i;
    int j;

    myfile=fopen("case2.txt", "r");

    for(i = 0; i < n; i++)
    {
        for (j = 0 ; j < m; j++)
        {
            fscanf(myfile,"%lf",A[i]+j);
        }
    }

    for(i = 0; i < n; i++)
    {
        for (j = 0 ; j < n; j++)
        {
            fscanf(myfile,"%lf",P[i]+j);
        }
    }

    for(i = 0; i < n; i++)
    {
        fscanf(myfile,"%lf",Q+i);
    }

    for(i = 0; i < m; i++)
    {
        fscanf(myfile,"%lf",u+i);
    }

    for (i = 0; i < m; i++)
        l[i] = 0 - INF;
    fclose(myfile);
}

void inverseDiag(CSR_d *A, CSR_d *B){
    copyCSR_d(A,B);
    for (int i =0; i < A->nnz; ++i)
    {
        B->d_val[i] = 1 / A->d_val[i];
    }
}



// fill a diagonal n*n matrix A with val on each diagonal entry
void initializeDiagMat(int n, double val, CSR_d *A){
    A->n = n;
    A->m = n;
    A->nnz = n;
    double d_val[n];
    int d_rowPtr[n+1], d_colInd[n];
    for (int i = 0; i < n; ++i) {
        d_val[i] = val;
        d_rowPtr[i] = i;
        d_colInd[i] = i;
    }
    d_rowPtr[n] = n;
    A->d_rowPtr = d_rowPtr;
    A->d_colInd = d_colInd;
    A->d_val = d_val;
    if(A->nnz > 0) {
        checkCusparseErrors(cusparseCreateCsr(&A->descr,
                                              n,
                                              n,
                                              n,
                                              d_rowPtr,
                                              d_colInd,
                                              d_val,
                                              CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_32I,
                                              CUSPARSE_INDEX_BASE_ZERO,
                                              CUDA_R_64F));
    }
}

//// min(l) max(u) projection of an vector A to B
//void vecMinMaxProj(int n, double A[n], double l[n], double u[n], double B[n]){
//    for (int i = 0; i < n; ++i)
//        if (A[i] < l[i])
//            B[i] = l[i];
//        else if (A[i] > u[i])
//            B[i] = u[i];
//        else
//            B[i] = A[i];
//}


//double norm2(VEC_d *A){
//    double norm = 0.0;
//    double squared_sum = 0.0;
//    for (int i = 0; i < A->n; ++i)
//        squared_sum += A->d_val[i] * A->d_val[i];
//    norm = sqrt(squared_sum/A->n);
//    return norm;
//}


double normInf(VEC_d *A){
    int idx;
    double h_res;
    checkCublasErrors(cublasIdamax(cublasHandle, A->n, A->d_val, 1, &idx));
    checkCudaErrors(cudaMemcpy(&h_res, A->d_val + (idx-1), sizeof(double), cudaMemcpyDeviceToHost));
    h_res = abs(h_res);
    return h_res;
}


//double normInf(VEC_d *A){
//    double maxValue = -INF;
//    for (int i = 0; i < A->n; ++i)
//        if (A->d_val[i] > maxValue)
//            maxValue = A->d_val[i];
//    return maxValue;
//}


// if l==u, penalty = 1000*rho, otherwise penalty = rho
void calculateR(int n, double *R, double *l, double *u, double rho){
    memset(R, 0, sizeof(R));
    for (int i = 0; i < n; ++i)
    {
        if (l[i] == u[i])
        {
            R[i*n+i] = rho * 1000;
        }
        else
        {
            R[i*n+i] = rho;
        }
    }
}

//get diagonal element of K into M
void getDiagonal(CSR_d *K, CSR_d *M)
{
    M->n = K->n;
    M->m = K->m;
    int nnz = 0;
    double d_val[K->n];
    int d_rowPtr[K->n+1], d_colInd[K->n];
    for (int i = 0; i < K->n; ++i) {
        int row_begin = K->d_rowPtr[i];
        int row_end = K->d_rowPtr[i+1];
        for (int j = row_begin; j < row_end; ++j) {
            if (K->d_colInd[j] == i) {
                d_val[nnz] = K->d_val[j];
                d_rowPtr[nnz] = j;
                d_colInd[nnz] = j;
                nnz++;
            }
        }
    }
    d_rowPtr[nnz+1] = nnz;
    M->d_rowPtr = d_rowPtr;
    M->d_colInd = d_colInd;
    M->d_val = d_val;
    if(nnz > 0) {
        checkCusparseErrors(cusparseCreateCsr(&M->descr,
                                              K->n,
                                              K->m,
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

// calculate objective value obj = 1/2 * x^T * P * x + q^T * X
double objValue(int n, CSR_d *P, VEC_d *q, VEC_d *x)
{
    VEC_d *temp = (VEC_d *) malloc(sizeof(VEC_d));
    vecMulMat(x,P,temp);
    double obj = 0.5 * innerProduct(temp, x) + innerProduct(q, x);
    return obj;
}

// calculate if the program should terminate
int termination(VEC_d *x, VEC_d *y, VEC_d *z, CSR_d *P, VEC_d *Q, CSR_d *A, CSR_d *AT, double epsilonPrimal, double epsilonDual){
    // temp1 = A*x
    VEC_d *temp1 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp2 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *residualPrimal = (VEC_d *) malloc(sizeof(VEC_d));

    //calculate residual of Primal
    matMulVec(A, x, temp1); // temp1 = A*x;
    scalarMulVec(-1, z, temp2); // temp2 = -z;
    vecAdd(temp1, temp2, residualPrimal); // residualPrimal = A*x - z;

    //calculate residual of Dual
    VEC_d *temp3 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp4 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp5 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *residualDual = (VEC_d *) malloc(sizeof(VEC_d));

    matMulVec(P, x, temp3); // temp3 = P*x;
    matMulVec(AT, y, temp4); // temp4 = A^T * y;
    vecAdd(temp3, temp4, temp5); // temp5 = temp3 + temp4;
    vecAdd(temp5, Q, residualDual); // residualDual = temp5 + Q;
    if ((normInf(residualPrimal) <= epsilonPrimal) && (normInf(residualDual) <= epsilonDual))
        return 1;
    else
        return 0;
}


// algorithm 3 with GPU
void solveKKT(int n, int m, VEC_d *x, VEC_d *y, VEC_d *z, CSR_d *P, VEC_d *Q, CSR_d *A, CSR_d *AT, VEC_d *l, VEC_d *u, CSR_d *R, VEC_d *xNext, VEC_d *zNext, double rho, double sigma, double epsilon)
{
    CSR_d *K = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_d *I = (CSR_d *) malloc(sizeof(CSR_d));
    initializeDiagMat(n, 1, I);
    CSR_d *temp1 = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_d *temp2 = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_d *temp3 = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_d *temp4 = (CSR_d *) malloc(sizeof(CSR_d));
    matMulMat(AT, R, temp1); // temp1 = A^T * R;
    matMulMat(temp1, A, temp2); // temp2 = A^T * R * A;
    scalarMulMat(sigma, I, temp3);
    matAdd(temp2, temp3, temp4);
    matAdd(temp4, P, K);

    CSR_d *M = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_d *MINV = (CSR_d *) malloc(sizeof(CSR_d));
    getDiagonal(K,M);
    inverseDiag(M,MINV);

    VEC_d *b = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp5 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp6 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp7 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp8 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp9 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp10 = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *temp11 = (VEC_d *) malloc(sizeof(VEC_d));
    matMulVec(R, z, temp5);
    scalarMulVec(-1, y, temp6);
    vecAdd(temp5, temp6, temp7); // temp7 = Rz - y;
    matMulVec(AT, temp7, temp8); // temp8 = A^T(Rz - y);
    scalarMulVec(sigma, x, temp9); // temp9 = sigma * x;
    scalarMulVec(-1, Q, temp10); // temp10 = -q;
    vecAdd(temp9, temp10, temp11); // temp11 = sigma * x - q;
    vecAdd(temp11, temp8, b); // b = sigma * x - q + A^T(Rz - y);

    //initialize r0=Kx-b, y0, p0
    VEC_d *r = (VEC_d *) malloc(sizeof(VEC_d));
    matMulVec(K, x, temp8); // temp8 = Kx;
    scalarMulVec(1, b, temp9); // temp9 = -b;
    vecAdd(temp8, temp9, r);

    //intitialize y0 = M^(-1)*r;
    VEC_d *y_kkt = (VEC_d *) malloc(sizeof(VEC_d));
    matMulVec(MINV, r, y_kkt); // y0 = M^(-1)*r

    // intitialize p0 = -y0;
    VEC_d *p = (VEC_d *) malloc(sizeof(VEC_d));
    scalarMulVec(-1, y_kkt, p);
    int k = 0;
    double normR = normInf(r);
    double normB = normInf(b);
    while (normR > epsilon * normB)
    {
        //calculate a^k
        double alpha;
        double temp12, temp13;
        temp12 = innerProduct(r, y_kkt); // calculate Numerator r^T * y;
        vecMulMat(p, K, temp8); // temp8 = p^T * K
        temp13 = innerProduct(temp8, p); // calculate denominator p^T * K * p;
        alpha = temp12 / temp13;
        //calculate x^k+1
        scalarMulVec(alpha, p, temp8);
        if (k==0)
            vecAdd(x, temp8, xNext); // x = x + alpha * p;
        else
            vecAddInPlace(xNext, temp8);
        //calculate r^k+1
        matMulVec(K, p,temp8); // temp8 = K * p;
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
    }
    matMulVec(A, xNext, zNext);
}

//void twoD2oneD(int n, int m, double* A, double *B)
//{
//    for (int i = 0; i < n; i++)
//        for (int j = 0; j < m; j++)
//            B[i*n+j] = *((A+i*n) + j);
//}
int main() {
    //
    int n,m;
    double sigma = 0.000001, alpha=1.6, rho = 0.5;

    //get n and m
    // input
//    //case 1
    n = 2;
    m = 2;
//  case 2
//    n = 10;
//    m = 10;
    double P_h[n*n],Q_h[n], A_h[m*n], AT_h[n*m], l_h[m],u_h[m];
    // case 1
    P_h[0] = 0.01;
    P_h[0] = 0.0;
    P_h[2] = 0.0;
    P_h[3] = 0.2889654;
    Q_h[0] = -1.07296862;
    Q_h[1] = 0.86540763;
    A_h[0] = 0.0;
    A_h[1] = 0.0;
    A_h[2] = 0.0;
    A_h[3] = -2.3015387;
    AT_h[0] = 0.0;
    AT_h[1] = 0.0;
    AT_h[2] = 0.0;
    AT_h[3] = -2.3015387;
    l_h[0] = 0 - INF;
    l_h[1] = 0 - INF;
    u_h[0] = 0.22957721;
    u_h[1] = -2.11756839;
 //   get_input(n,m,P,Q,l,u,A);
    //get AT = A^T
    //transpose(m,n, A_h, AT_h);
    //initialize x,y,z
    double R_h[m*m];
    calculateR(m, R_h, l_h, u_h, rho);
    //double A_2d[m*n];
    //double AT_2d[n*m];
    //double P_2d[n*n];

    DN_h *A_dh_h = (DN_h *) malloc(sizeof(DN_h));
    //twoD2oneD(m, n, A_h, A_2d);
    initDN_h(A_dh_h, m, n, A_h);
    CSR_h *A_csrh = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(A_dh_h, A_csrh);
    CSR_d *A = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(A_csrh, A);

    DN_h *AT_dh_h = (DN_h *) malloc(sizeof(DN_h));
    //twoD2oneD(n, m, AT_h, AT_2d);
    initDN_h(AT_dh_h, n, m, AT_h);
    CSR_h *AT_csrh = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(AT_dh_h, AT_csrh);
    CSR_d *AT = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(AT_csrh, AT);

    DN_h *P_dh_h = (DN_h *) malloc(sizeof(DN_h));
    //twoD2oneD(n, n, P_h, P_2d);
    initDN_h(P_dh_h, n, n, P_h);
    CSR_h *P_csrh = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(P_dh_h, P_csrh);
    CSR_d *P = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(P_csrh, P);

    DN_h *R_dh_h = (DN_h *) malloc(sizeof(DN_h));
    initDN_h(R_dh_h, m, m, R_h);
    CSR_h *R_csrh = (CSR_h *) malloc(sizeof(CSR_h));
    DN_h2CSR_h(R_dh_h, R_csrh);
    CSR_d *R = (CSR_d *) malloc(sizeof(CSR_d));
    CSR_h2d(R_csrh, R);

    CSR_d *RINV = (CSR_d *) malloc(sizeof(CSR_d));

    inverseDiag(R, RINV);


    double x_h[n], y_h[m], z_h[m];
    memset(x_h, 0, sizeof(x_h));
    memset(y_h, 0, sizeof(y_h));
    memset(z_h, 0, sizeof(z_h));
    VEC_d *x = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *y = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *z = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *l = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *u = (VEC_d *) malloc(sizeof(VEC_d));
    VEC_d *Q = (VEC_d *) malloc(sizeof(VEC_d));

    initVEC_d(x, n, x_h);
    initVEC_d(y, m, y_h);
    initVEC_d(z, m, z_h);
    initVEC_d(l, m, l_h);
    initVEC_d(u, m, u_h);
    initVEC_d(Q, m, Q_h);

    // initialize sigma and alpha
    // initialize epsilon, now we assume epsilon is a constant
    // todo add a function to calculate epsilon in each iteration;
    double epsilon = 0.00001;
    double epsilonPrimal = 0.0001;
    double epsilonDual = 0.0001;
    //eps = calc_eps();
    int k = 0;



    while (!termination(x, y, z, P, Q, A, AT, epsilonPrimal, epsilonDual) && k <= 100)
    {
        VEC_d *xNext = (VEC_d *) malloc(sizeof(VEC_d));
        VEC_d *zNext = (VEC_d *) malloc(sizeof(VEC_d));

        //calculate the penalty matrix R and its inverse RINV

        solveKKT(n, m, x, y, z, P, Q, A, AT, l,u, R, xNext, zNext, rho, sigma, epsilon);
        // update x
        VEC_d *temp1 = (VEC_d *) malloc(sizeof(VEC_d));
        VEC_d *temp2 = (VEC_d *) malloc(sizeof(VEC_d));

        scalarMulVec(alpha, xNext, temp1); // temp1 = alpha * xNext;
        scalarMulVec(1 - alpha, x, temp2); // temp2 = (1 - alpha) * x;
        vecAdd(temp1, temp2, x); // x = alpha * xNext +  (1 - alpha) * x;

        // update z
        VEC_d *temp3 = (VEC_d *) malloc(sizeof(VEC_d));
        VEC_d *temp4 = (VEC_d *) malloc(sizeof(VEC_d));
        VEC_d *temp5 = (VEC_d *) malloc(sizeof(VEC_d));
        VEC_d *temp6 = (VEC_d *) malloc(sizeof(VEC_d));
        VEC_d *temp7 = (VEC_d *) malloc(sizeof(VEC_d));
        VEC_d *zNextReal = (VEC_d *) malloc(sizeof(VEC_d));
        scalarMulVec(alpha, zNext, temp3); // temp3 = alpha * zNext;
        scalarMulVec(1 - alpha, z, temp4); // temp4 = (1 - alpha) * z;
        matMulVec(RINV,y, temp5 ); // temp5 = R^(-1) * y;
        vecAdd(temp3, temp4, temp6);  // temp6 = alpha * zNext + (1 - alpha) * z;
        vecAdd(temp5, temp6, temp7);    // temp7 = alpha * zNext + (1 - alpha) * z + R^(-1) * y
        vecMinMaxProj(l->n, temp7, l, u, zNextReal); // zNextReal is the projection of temp7

        // update y
        scalarMulVec(-1, zNextReal, temp3); // temp3 = -zNextReal;
        vecAdd(temp6, temp3, temp4); // temp4 = alpha * zNext + (1 - alpha) * z - zNextReal;
        matMulVec(R, temp4, temp5); // temp5 = R * (alpha * zNext + (1 - alpha) * z - zNextReal);
        vecAddInPlace(y, temp5); // y = y + temp5;

        // update z^k to z^(k+1)
        z = zNextReal;
        printf("Round:%d, x1:%.6f, x2:%.6f, obj:%.6f\n", k, x->d_val[0], x->d_val[1], objValue(n,P,Q,x));
        k += 1;
    }
    return 0;
}
