#include <iostream>


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

// return inner project of two vectors
double innerProduct(int n, double A[n], double B[n]){

}


// calculate sum of two vectors
void vecAdd(int n, int A[n], double B[n], double C[n]){

}

// min(l) max(u) projection of an vector
void vecMinMaxProj(int n, double A[n], double l[n], double u[n], double B[n]){


}

// B = val * A
void scalarMulVec(int n, double val, double A[n], double B[n])
{


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
void solve_kkt(int n, int m, double x[n], double y[m], double z[m], double P[n][n], double Q[n], double A[m][n], double AT[n][m], double l[m], double u[m], double R[m][m], double xNext[n], double zNext[m], double rho, double sigma, double epsilon){
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
    scalarMulVec(n, -1, q, temp10); // temp10 = -q;
    vecAdd(n, temp9, temp10, temp11); // temp11 = sigma * x - q;
    vecAdd(n, temp11, temp8, b); // b = sigma * x - q + A^T(Rz - y);

    //initialize r0=Kx-b, y0, p0
    double r[n];
    MatMulVec(n, n, K, x, temp8); // temp8 = Kx;
    scalarMulVec(n, -1, b, temp9); // temp9 = -b;
    vecAdd(n, temp8, temp9, r);

    //intitialize y0 = M^(-1)*r;
    double y[n];
    diagMatMulVec(n,n, MINV, r, y); // y0 = M^(-1)*r

    // intitialize p0 = -y0;
    double p[n];
    scalarMulVec(n, -1, y, p);
    k = 0;

    while (norm2(r) > epsilon * norm2(b))
    {
        //calculate a^k
        double alpha;
        double temp12, temp13;
        temp12 = innerProduct(n, r, y); // calculate Numerator r^T * y;
        vecMulMat(n,n, p, K, temp8); // temp8 = p^T * K
        temp13 = innerProduct(n,temp8, p); // calculate denominator p^T * K * p;
        alpha = temp12 / temp13;
        //calculate x^k+1
        scalarMulVec(n, alpha, p, temp8);
        vecAdd(n, alpha, temp8, alpha); // x = x + alpha * p;
        //calculate r^k+1
        double r[n];
        diagMatMulVec(n, n, K, p,temp8); // temp8 = K * p;
        scalarMulVec(n,alpha, temp8, temp9); // temp9 = alpha * K * p;
        vecAdd(n, r, temp9, r);
        //calculate y^k+1
        double yNew[n];
        diagMatMulVec(n, n, MINV, r, y); // y = M^(-1) * rNew;
        //calculate beta^k+1
        double beta;
        temp13 = innerProduct(n, r, y); // calculate Numerator rNew^T * yNew;
        beta = temp13 / temp12; // beta = ( rNew^T * yNew) / ( r^T * y)
        //calculate p^k+1
        scalarMulVec(n, -1, y, temp8); // temp8 = -y
        scalarMulVec(n, beta, p, temp9); // temp9 = beta * p;
        vecAdd(n, temp8, temp9, p);
        k+=1;
    }
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
    double epsilon = 0.00001
    //eps = calc_eps();
    int k = 0;
    while (!termination(n, m, x, y, z, P, Q, A, AT, eps))
    {
        double xNext[n], zNext[n];
        double R[m][m], RINV[m][m];

        //calculate the penalty matrix R and its inverse RINV
        calculateR(R);
        inverseDiag(m, R, RINV);

        solve_kkt(n, m, x, y, z, P, Q, A, AT, l,u, R, xNext, zNext, rho, sigma, epsilon);
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
        matMulVec(m, m, R, temp4, temp5) // temp5 = R * (alpha * zNext + (1 - alpha) * z - zNextReal);
        vecAdd(m, y, temp5, y) // y = y + temp5;

        // update z^k to z^(k+1)
        z = zNextReal // z = zNextReal
        k += 1;
    }
    return 0;
}
