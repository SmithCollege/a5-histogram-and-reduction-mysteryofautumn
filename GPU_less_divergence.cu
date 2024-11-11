#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>

#define SIZE 16384
#define BLOCK_SIZE 1024

double get_clock() {
    struct timeval tv; 
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok<0) { printf("get time ofday error"); }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

__device__ float __sum(float i, float j){
    return i + j;
}

__device__ float __product(float i, float j){
    return i*j;
}

__device__ float __min(float i, float j){
    if (i<j){return i;}
    else{return j;}
}

__device__ float __max(float i, float j){
    if (i>j){return i;}
    else{return j;}
}

__global__ void reduction(float* M, float* partialResults, int operation) {
    __shared__ float shared[2*BLOCK_SIZE];

    unsigned int index = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    if (start + index < SIZE) {
        shared[index] = M[start + index];
        shared[blockDim.x + index] = M[start + blockDim.x + index];
    }

    for (unsigned int stride = blockDim.x; stride >= 1; stride >>= 1){

        __syncthreads();
        
        if (index < stride){
            switch (operation){
            case 1: //sum
                shared[index] = __sum(shared[index], shared[index + stride]);
                break;
            case 2: //product
                shared[index] = __product(shared[index],shared[index + stride]);
                break;
            case 3: //max
                shared[index] = __max(shared[index], shared[index + stride]);
                break;
            case 4: //min
                shared[index] = __min(shared[index], shared[index + stride]);
                break;
            }
        }

    }

    if (index == 0) {
        partialResults[blockIdx.x] = shared[0];
    }
}



int main(void)
{
    float* M, *partialResult;
    int numBlocks = (SIZE + BLOCK_SIZE - 1) / (2*BLOCK_SIZE);
    int operation = 1;


    cudaMallocManaged(&M, sizeof(float) * SIZE);
    cudaMallocManaged(&partialResult, sizeof(float) * numBlocks);


    // initialization
    for (int i = 0; i < SIZE; i++) {
        M[i] = 1;
    }

    double t0 = get_clock();

    // Run kernel 

    reduction<<<numBlocks, BLOCK_SIZE>>>(M, partialResult, operation);

  
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    double t1 = get_clock();

    float total = 0.0f;
    
    for (int i = 0; i < numBlocks; i++) {
        switch (operation){
            case 1: //sum
                total += partialResult[i];
                break;
            case 2: //product
                total *= partialResult[i];
                break;
            case 3: //max
                total = max(total, partialResult[i]);
                break;
            case 4: //min
                total = min(total, partialResult[i]);
                break;
        }
    }


    printf("size: %d\n", SIZE);
    printf("time: %f ns\n", (1000000000.0 * (t1 - t0)));


    // Error checking 
    printf("total: %f\n", total);

    // Free memory
    cudaFree(M);  
    cudaFree(partialResult);
    return 0;
}