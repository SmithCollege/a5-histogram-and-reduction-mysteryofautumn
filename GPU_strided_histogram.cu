#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#define SIZE 1000000
#define BLOCK_SIZE 1000

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { printf("get time ofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__global__ void histogram(int* M, int* N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < SIZE){
        atomicAdd(&N[(M[index]%10)-1],1);
    }

}


int main(void)
{
    int* M, *N;

    cudaMallocManaged(&M, sizeof(int) * SIZE);
    cudaMallocManaged(&N, sizeof(int) * 10);

    // initialization
    for (int i = 0; i < SIZE; i++) {
        M[i] = 1;
    }

    int numBlocks = (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double t0 = get_clock();

    // Run kernel 

    histogram<<<numBlocks, BLOCK_SIZE>>>(M, N);
  
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    double t1 = get_clock();
    printf("size: %d\n", SIZE);
    printf("time: %f ns\n", (1000000000.0 * (t1 - t0)));


    // Error checking 
    printf("N[0]: %d\n", N[0]);

    // Free memory
    cudaFree(M);  
    return 0;
}