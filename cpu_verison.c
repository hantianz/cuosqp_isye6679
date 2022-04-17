//
// Created by Hantian Zhang on 4/4/2022.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INF (1 << 30)

void get_input(int n, int m, double P[n][n], double Q[n], double l[m], double u[m], double A[m][n]){
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


// calculate inverse of a diagonal matrix A and store it in matrix B
void inverseDiag(int n, double A[n][n], double B[n][n]){
    memset(B, 0, sizeof(B));
    for (int i =0; i < n; ++i)
    {
        B[i][i] = 1.0 / A[i][i];
    }
}

// calculate transpose of a matrix A and store it in matrix B
void transpose(int n, int m, double A[n][m], double B[m][n]){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            B[j][i] = A[i][j];
}

// fill a diagonal n*n matrix A with val on each diagonal entry
void initializeDiagMat(int n, double val, double A[n][n]){
    memset(A, 0, sizeof(A));
    for (int i = 0; i < n; ++i)
        A[i][i] = val;
}

// calculate A*B and store it in C
void matMulMat(int n, int m, int t, double A[n][m], double B[m][t], double C[n][t]){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < t; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < m; ++k)
                C[i][j] += A[i][k] * B[k][j];
        }
}

// calculate A*B and store it in C, B is diagonal
void matMulDiagMat(int n, int m, double A[n][m], double B[m][m], double C[n][m]){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
        {
            C[i][j] = A[i][j] * B[j][j];
        }
}

// calculate A*B and store it in C, A is diagonal
void diagMatMulMat(int n, int m, double A[n][n], double B[n][m], double C[n][m]){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
        {
            C[i][j] = B[i][j] * A[i][i];
        }
}

// calculate matrix A* vector B and store it in vector C, A is diagonal
void diagMatMulVec(int n, int m, double A[n][n], double B[n], double C[n]){
    for (int i = 0; i < n; ++i)
    {
        C[i] = A[i][i] * B[i];
    }
}

// calculate vector A* matrix B and store it in vector C
void vecMulDiagMat(int n, double A[n], double B[n][n], double C[n]){
    for (int i = 0; i <n; ++i)
        C[i] = A[i] * B[i][i];
}



// calculate matrix A* vector B and store it in vector C
void matMulVec(int n, int m, double A[n][m], double B[m], double C[n]){
    for (int i = 0; i < n; ++i)
    {
        C[i] = 0;
        for (int j = 0; j < m; ++j)
            C[i] += A[i][j] * B[j];
    }

}

// calculate vector A* matrix B and store it in vector C
void vecMulMat(int n, int m, double A[n], double B[n][m], double C[m]) {
    for (int i = 0; i < m; ++i) {
        C[i] = 0;
        for (int j = 0; j < n; ++j)
            C[i] += A[j] * B[j][i];
    }
}
// calculate sum of two matrices
void matAdd(int n, int m, double A[n][m], double B[n][m], double C[n][m]){
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i][j] = A[i][j] + B[i][j];
}

// return inner project of two vectors
double innerProduct(int n, double A[n], double B[n]){
    double product = 0.0;
    for (int i = 0; i < n; ++i)
        product += A[i] * B[i];
    return product;
}


// calculate sum of two vectors
void vecAdd(int n, double A[n], double B[n], double C[n]){
    for (int i = 0; i < n; ++i)
        C[i] = A[i] + B[i];
}

// min(l) max(u) projection of an vector A to B
void vecMinMaxProj(int n, double A[n], double l[n], double u[n], double B[n]){
    for (int i = 0; i < n; ++i)
        if (A[i] < l[i])
            B[i] = l[i];
        else if (A[i] > u[i])
            B[i] = u[i];
        else
            B[i] = A[i];
}

// B = val * A
void scalarMulVec(int n, double val, double A[n], double B[n])
{
    for (int i = 0; i < n; ++i)
        B[i] = A[i] * val;
}

// B = val * A
void scalarMulDiagMat(int n, double val, double A[n][n], double B[n][n])
{
    for (int i = 0; i < n; ++i)
        B[i][i] = A[i][i] * val;
}

// calculate norm of a vector
double norm2(int n, double A[n]){
    double norm = 0.0;
    double squared_sum = 0.0;
    for (int i = 0; i < n; ++i)
        squared_sum += A[i] * A[i];
    norm = sqrt(squared_sum/n);
    return norm;
}

double normInf(int n, double A[n]){
    double norm = 0.0;
    double maxValue = -1;
    for (int i = 0; i < n; ++i) {
        if (fabs(A[i]) > maxValue) {
            maxValue = fabs(A[i]);
        }
    }
    norm = maxValue;
    return norm;
}

// if l==u, penalty = 1000*rho, otherwise penalty = rho
void calculateR(int n, double R[n][n], double l[n], double u[n], double rho){
    memset(R, 0, sizeof(R));
    for (int i = 0; i < n; ++i)
        if (l[i] == u[i])
            R[i][i] = rho * 1000;
        else
            R[i][i] = rho;
}

//get diagonal element of K into M
void calculatePrecond(int n, double K[n][n], double M[n][n]){
    memset(M, 0, sizeof(M));
    for (int i = 0; i < n; ++i)
        M[i][i] = K[i][i];
}

// calculate objective value obj = 1/2 * x^T * P * x + q^T * X
double objValue(int n, double P[n][n], double q[n], double x[n])
{
    double temp[n];
    vecMulMat(n,n,x,P,temp);
    double obj = 0.5 * innerProduct(n, temp, x) + innerProduct(n, q, x);
    return obj;
}

// calculate if the program should terminate
int termination(int n, int m, double x[n], double y[m], double z[m], double P[n][n], double Q[n], double A[m][n], double AT[n][m], double epsilonPrimal, double epsilonDual){
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

    // printf("%f, %f\n", normInf(m, residualPrimal), epsilonPrimal);

    if ((normInf(m, residualPrimal) <= epsilonPrimal) && (normInf(n, residualDual) <= epsilonDual))
        return 1;
    else
        return 0;
}


// algorithm 3 solve kkt with PCG
void solveKKT(int n, int m, double x[n], double y[m], double z[m], double P[n][n], double Q[n], double A[m][n], double AT[n][m], double l[m], double u[m], double R[m][m], double xNext[n], double zNext[m], double rho, double sigma, double epsilon){
    //todo Add a function to update rho in every 10 iterations
    double K[n][n], I[n][n];
    //calculate K = P + sigmaI + A^T*R*A, K is symmetric positive definite matrix;
    // initialize an identity matrix I
    initializeDiagMat(n, 1, I);
    double temp1[n][m], temp2[n][n], temp3[n][n], temp4[n][n];
    matMulDiagMat(n,m,AT, R, temp1); // temp1 = A^T * R;
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
    double normR = normInf(n,r);
    double normB = normInf(n,b);



    // printf("r");
    // printDVec(n, r);
    // printf("normR :%f\n", normR);
    // printf("normB :%f\n", normB);


    while (normR > epsilon * normB)
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
        // printf("p:");
        // printDVec(n, p);

        if (k==0)
            vecAdd(n, x, temp8, xNext); // x = x + alpha * p;
        else
            vecAdd(n, xNext, temp8, xNext);
        //calculate r^k+1
        double rNew[n];
        diagMatMulVec(n, n, K, p,temp8); // temp8 = K * p;
        scalarMulVec(n,alpha, temp8, temp9); // temp9 = alpha * K * p;
        vecAdd(n, r, temp9, r);

        // printf("temp8:");
        // printDVec(n, temp8);   

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
        normR = normInf(n,r);

        // printf("normR :%f\n", normR);
        // printf("normB :%f\n", normB);
    }
    matMulVec(m,n, A, xNext, zNext);
}

void printDVec(int n, double *A) {
    for (int i = 0; i < n; i++) {
        printf("%f, ", A[i]);
    }
    printf("\n");
}

void printIVec(int n, int *A) {
    for (int i = 0; i < n; i++) {
        printf("%d, ", A[i]);
    }
    printf("\n");
}

void printMatrix(int m, int n, double *A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f,", A[i*n+j]);
        }
        printf("\n");
    }
}


int main() {
    //
    int n,m;
    //get n and m
    // input
   //case 1
   n = 2;
   m = 2;
//  case 2
    // n = 10;
    // m = 10;
    double P[n][n],Q[n], A[m][n], AT[n][m], l[m],u[m];
//    // case 1
   P[0][0] = 0.01;
   P[0][1] = 0.0;
   P[1][0] = 0.0;
   P[1][1] = 0.2889654;
   Q[0] = -1.07296862;
   Q[1] = 0.86540763;
   A[0][0] = 0.0;
   A[0][1] = 0.0;
   A[1][0] = 0.0;
   A[1][1] = -2.3015387;
   l[0] = 0 - INF;
   l[1] = 0 - INF;
   u[0] = 0.22957721;
   u[1] = -2.11756839;
    // get_input(n,m,P,Q,l,u,A);
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
    double epsilonPrimal = 0.0001;
    double epsilonDual = 0.0001;
    //eps = calc_eps();
    int k = 0;

    while (!termination(n, m, x, y, z, P, Q, A, AT, epsilonPrimal, epsilonDual) && k <= 100)
    {
        double xNext[n], zNext[n];
        double R[m][m], RINV[m][m];
        //calculate the penalty matrix R and its inverse RINV
        calculateR(m, R, l, u, rho);
        inverseDiag(m, R, RINV);

        solveKKT(n, m, x, y, z, P, Q, A, AT, l,u, R, xNext, zNext, rho, sigma, epsilon);

        // printDVec(n, xNext);

        // update x
        double temp1[n], temp2[n];
        scalarMulVec(n, alpha, xNext, temp1); // temp1 = alpha * xNext;
        scalarMulVec(n, 1 - alpha, x, temp2); // temp2 = (1 - alpha) * x;
        vecAdd(n, temp1, temp2, x); // x = alpha * xNext +  (1 - alpha) * x;

        // update z
        double temp3[m], temp4[m], temp5[m], temp6[m], temp7[m], zNextReal[m];
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
        memcpy(z, zNextReal, sizeof(zNextReal)); // z = zNextReal
        printf("Round:%d, x1:%.6f, x2:%.6f, obj:%.6f\n", k, x[0], x[1], objValue(n,P,Q,x));
        k += 1;

    }
    return 0;
}
