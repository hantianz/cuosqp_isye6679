// Euclidean projection onto [l, u] of vector A: min(max(x, l), u)
__global__ void vecMinMaxProj(int n,
                              float *A,
                              float *l,
                              float *u,
                              float *B
                              ){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n) {
        B[i] = c_min(u[i], c_max(l[i],A[i]));
    }
}

// Calculate inverse of vector A and store it in vector B
__global__ void inverseVec(float *A,
                           float *B){

    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < n) {
        B[i] = 1.0 / A[i];
    }
}
