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

_global__ void stencil(float *a,float *b,int m,int n){
    __shared__ float mem[SMEM_W][SMEM_W];
    int row_off = blockIdx.y * blockDim.y;
    int col_off = blockIdx.x * blockDim.x;
    int gy = row_off - R;
    int gx = col_off - R;
    int total = SMEM_W * SMEM_W;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int grid = blockDim.y * blockDim.x;
    for(int i=tid;i<total;i+=grid){
        int lx = tid & SMEM_W;
        int ly = tid / SMEM_W;
        int gy0 = ly + gy;
        int gx0 = lx + gx;
        gy0 = min(max(0,gy0),m);
        gx0 = min(max(0,gx0),n);
        memp[ly][lx] = a[m]
    }
    if((row_off + threadIdx.y) < m && (col_off + threadIdx.x) < n){
        b[(row_off + threadIdx) * n + col_off + threadIdx] =
                        mem[threadIdx.y][threadIdx.x] *0.5f
                    +   mem[threadIdx.y-1][threadIdx.x] * 0.125f
                    +   mem[threadIdx.y+1][threadIdx.x] * 0.125f
                    +   mem[threadIdx.y][threadIdx.x-1] * 0.125f
                    +   mem[threadIdx.y][threadIdx.x+1] * 0.125f
    }
}

int main(){
    int m =9;
    int n= 9;
    float *a =(float *)malloc(sizeof(float) * m * n);
    float *b =(float *)malloc(sizeof(float) * m * n);
    Init(a,m * n);
    Init(b,m * n,false);
    
    stencil_clamp_cpu(a,b,m,n);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            std::cout<<a[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            std::cout<<b[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
}