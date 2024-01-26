
//nvcc -rdc=true ./SinTest.cu -o test
//nsys profile ./test

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <chrono>

#define LOOP_NUM 10

using namespace std::chrono;

__device__  void sinKernel(double *a, double *b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = sin(a[i]);
}

__global__ void Sin_1(double *a, double *b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<268435456){
        sinKernel(a, b);
    }
}

__global__ void Sin_2(double *a, double *b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<268435456){
        for(int j=0;j<LOOP_NUM;j++){
            sinKernel(a, b);
            __syncthreads();
        }
    }
}

int main(){

	const int N = 1024*1024*256;
  
	double *a = (double*)malloc(N*sizeof(double));
    double *b = (double*)malloc(N*sizeof(double));
	

	for(int i=0;i<N;i++){
		a[i]=i;
	}
    
    double *dev_a = 0;
    double *dev_b = 0;
 
    cudaMalloc((void**)&dev_a, N * sizeof(double));
    cudaMalloc((void**)&dev_b, N * sizeof(double));

    cudaMemcpy(dev_a, a, N * sizeof(double), cudaMemcpyHostToDevice);

    auto start1 = steady_clock::now();
    for(int i=0;i<LOOP_NUM;i++){
        Sin_1<<<262144, 1024>>>(dev_a, dev_b);  
    }
    cudaDeviceSynchronize();
    auto end1 = steady_clock::now();
    auto t1 = duration_cast<std::chrono::milliseconds>(end1 - start1);
 
    auto start2 = steady_clock::now();
    Sin_2<<<262144, 1024>>>(dev_a, dev_b);
    cudaDeviceSynchronize();
    auto end2 = steady_clock::now();
    auto t2 = duration_cast<std::chrono::milliseconds>(end2 - start2);

    cudaMemcpy(b, dev_b, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("time1: %dms time2: %dms\n", t1.count(), t2.count());
    cudaFree(dev_a);
    cudaFree(dev_b);
}

