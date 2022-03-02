#include <iostream>


void get_input(double ** P, double *  q, double* l, double * u, double ** A){


};

// calculate epsilon for termination
double calc_eps(){

}

// calculate inverse of a matrix A and store it in matrix B
void inverse(doubel ** A, double ** B){

}

// calculate transpose of a matrix A and store it in matrix B
void transpose(doubel ** A, double ** B){

}

// calculate A*B and store it in C
void matmul(doubel ** A, double ** B, double **C){

}

// calculate matrix A* vector B and store it in vector C
void matmulvec(doubel ** A, double * B, double *C){

}

// calculate norm of a vector
double norm(doubel *A){
    double norm = 0.0;
    return norm;
}

// calculate if the program should terminate
bool termination(x,y,z,p,q,eps){
  return 0;
}

// algorithm 3 solve kkt with PCG
void solve_kkt(x,M,K,b){
    //initialize r0, y0, p0
    r0 = K * X0 - b;
    y0 = inverse(M)
    k = 0;
    t = r0;
    while (norm(r) > epsilon * norm(b))
    {
        //calculate a^k
        //calculate x^k+1
        //calculate r^k+1
        //calculate y^k+1
        //calculate beta^k+1
        //calculate p^k+1
        k+=1
    }
}

void calculate_precond(M){

}

int main() {
    std::cout << "Hello, World!" << std::endl;
    // input
    get_input(P,Q,l,u,A);
    //initialize x,y,z
    double *x, *y, *z;
    double sigma = 0.5, alpha=0.5;
    eps = calc_eps();
    int k = 0;
    while (!termination(x,y,z,p,q,eps))
    {
        //calculate preconditioner M
        calculate_precond(M);
        solve_kkt(x,y,z,p,q);
        // update x
        // update z
        // update y
        k += 1;
    }
    return 0;
}
