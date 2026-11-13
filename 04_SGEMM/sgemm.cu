#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"../common/cuda_check.cuh"
#include"../common/reference.h"
#include"../common/timer.cuh"
#include"../common/benchmark.cuh"
#include<stdio.h>
#include<iostream>

#define TILE_Y 16
#define TILE_X 16

__global__ void gemm_naive_gpu(float *a,float *b,float *c,int m,int n,int k){
    int row_off = blockIdx.y *TILE_Y + threadIdx.y;
    int col_off = blockIdx.x *TILE_X + threadIdx.x;

    int idx = row_off * n + col_off;
    float sum = 0.0f;
    for(int i =0;i< k;i++){
        sum += a[row_off * k + i] * b[i * n + col_off];
    }
    c[idx] = sum;
}

void gemm_naive(int m,int n,int k){
    float *a = nullptr;
    float *b =nullptr;
    float * c = nullptr;
    float * comp =nullptr;
    int mk = m*k;
    int nk =n*k;
    int mn =m*n;
    cudaMallocHost(&a,sizeof(float) * mk);
    cudaMallocHost(&b,sizeof(float) * nk);
    cudaMallocHost(&c,sizeof(float) * mn);
    cudaMallocHost(&comp,sizeof(float) * mn);

    Init(a,mk);
    Init(b,nk);
    Init(c,mn,false);
    Init(comp,mn,false);

    float *dev_a = nullptr;
    float *dev_b = nullptr;
    float *dev_c = nullptr;
    cudaMemcpy(dev_a,a,sizeof(float) * mk,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,sizeof(float) * nk,cudaMemcpyHostToDevice);
    dim3 grid = dim3((m + TILE_Y -1)/TILE_Y,(n + TILE_X -1 )/TILE_X);
    dim3 block - dim3(TILE_Y,TILE_X);
    gemm_naive_gpu<<<grid,block>>>(dev_a,dev_b,dev_c,m,n,k);
    cudaMemcpy(c,dev_c,sizeof(float) * mn,cudaMemcpyDeviceToHost);

    if(equal(c,comp,m * n)){
		std::cout<<"Functional GEMM_NAIVE Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional GEMM_NAIVE Testing WRONG!!!"<<std::endl;
	}
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(comp);
}

int main(){
    int m = 512;
    int n =512;
    int k = 512;
    gemm_naive(m,n,k);
}