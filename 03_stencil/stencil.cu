#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"../common/cuda_check.cuh"
#include"../common/reference.h"
#include"../common/timer.cuh"
#include"../common/benchmark.cuh"
#include<stdio.h>
#include<iostream>

#define R 1
#define TILE_X 32
#define TILE_Y 32
#define SMEM_W  (TILE_X + 2*R)      
#define SMEM_WP (SMEM_W + 1)        

__global__ void stencil(float *a,float *b,int m,int n){
    __shared__ float mem[SMEM_W][SMEM_W];
    int row_off = blockIdx.y * blockDim.y;
    int col_off = blockIdx.x * blockDim.x;
    int gy = row_off - R;
    int gx = col_off - R;
    int total = SMEM_W * SMEM_W;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int grid = blockDim.y * blockDim.x;
    for(int i=tid;i<total;i+=grid){
        int lx = i % SMEM_W;
        int ly = i / SMEM_W;
        int gy0 = ly + gy;
        int gx0 = lx + gx;
        gy0 = min(max(0,gy0),m-1);
        gx0 = min(max(0,gx0),n-1);
        mem[ly][lx] = a[gy0 * n +gx0];
    }
    __syncthreads();
    int oy = row_off + threadIdx.y;
    int ox = col_off + threadIdx.x;
    if (oy < m && ox < n) {
        int cy = threadIdx.y + R;
        int cx = threadIdx.x + R;
        b[(size_t)oy * n + ox] =
              mem[cy][cx]     * 0.5f
            + mem[cy-1][cx]   * 0.125f
            + mem[cy+1][cx]   * 0.125f
            + mem[cy][cx-1]   * 0.125f
            + mem[cy][cx+1]   * 0.125f;
    }
}



void stencil_clamp(int m,int n){
    float *a = nullptr;
    float *b =nullptr;
    float *comp =nullptr;
    int length = m * n;
    cudaMallocHost(&a,sizeof(float) * length);
    cudaMallocHost(&b,sizeof(float) * length);
    cudaMallocHost(&comp,sizeof(float) * length);

    Init(a,m * n);
    Init(b,m * n,false);
    Init(comp,m * n,false);
    
    float *dev_a=nullptr;
    float *dev_b=nullptr;
    cudaMalloc(&dev_a,sizeof(float) * length);
    cudaMalloc(&dev_b,sizeof(float) * length);
    int grid_y = (m+ TILE_Y -1)/TILE_Y;
    int grid_x = (n+ TILE_X -1)/TILE_X;
    dim3 grid = dim3(grid_y,grid_x);
    dim3 block = dim3(TILE_Y,TILE_X);
    cudaMemcpy(dev_a,a,sizeof(float) * length,cudaMemcpyHostToDevice);
    stencil<<<grid,block>>>(dev_a,dev_b,m,n);
    cudaMemcpy(b,dev_b,sizeof(float) * length,cudaMemcpyDeviceToHost);
    stencil_clamp_cpu(a,comp,m,n);

    if(equal(b,comp,length)){
		std::cout<<"Functional STENCIL Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional STENCIL Testing WRONG!!!"<<std::endl;
	}
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            std::cout<<b[i * 3 +j]<<" ";
        }
        std::cout<<std::endl;
    }
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            std::cout<<comp[i*3 +j]<<" ";
        }
        std::cout<<std::endl;
    }
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(comp);
}



int main(){
    stencil_clamp(16,16);
}