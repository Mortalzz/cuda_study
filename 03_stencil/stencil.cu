#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"../common/cuda_check.cuh"
#include"../common/reference.h"
#include"../common/timer.cuh"
#include"../common/benchmark.cuh"
#include<stdio.h>
#include<iostream>



int main(){
    int m =9;
    int n= 9;
    float *a =(float *)malloc(sizeof(float) * m * n);
    float *b =(float *)malloc(sizeof(float) * m * n);
    Init(a,m * n);
    Init(b,m * n,false);
    
    stencil_cpu(a,b,m,n);
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