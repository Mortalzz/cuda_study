#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"../common/cuda_check.cuh"
#include"../common/reference.h"
#include"../common/timer.cuh"
#include"../common/benchmark.cuh"
#include<stdio.h>
#include<iostream>

#define BLOCK_SIZE 256
#define BLOCK_SIZE_LITE 32

__global__ void reduce(float *a, float *b,int length){
    __shared__ float mem[BLOCK_SIZE];
    int off = blockIdx.x * blockDim.x;
    int idx = threadIdx.x;
    mem[idx] = idx + off < length ? a[off + idx]:0.0f;
    __syncthreads();

    for(int i = BLOCK_SIZE / 2;i>0;i/=2){
        if(idx < i){
            mem[idx] += mem[idx + i];
        }
        __syncthreads();
    }

    if(idx ==0){
        b[blockIdx.x] = mem[0];
    }
}

__global__ void exclusive_scan_lite(float *a, float *b, int length) {
    __shared__ float temp[64];
    int idx = threadIdx.x;

    temp[idx] = (idx > 0 && idx < length) ? a[idx - 1] : 0.0f;
    __syncthreads();
    bool j =true;
    for (int stride = 1; stride < length; stride <<= 1) {
        if(idx < length){
            if(j){
                temp[32 + idx] =idx >= stride?temp[idx - stride] + temp[idx]:temp[idx];
            }else{
                temp[idx] = idx >= stride? temp[32 + idx -stride] + temp[32 + idx] : temp[idx + 32];
            }
            j=!j;
        }
        __syncthreads();  // 保证本轮写完再进入下一轮
    }
    __syncthreads();
    if (idx < length)
        b[idx] = j?temp[idx]:temp[idx+32];
}

__global__ void exclusive_scan_block(float *a ,float *lite,float *b,int length){
    __shared__ float mem[BLOCK_SIZE * 2];

    int off = blockIdx.x * blockDim.x ;
    int idx = threadIdx.x;

    mem[idx] = (idx > 0 && idx + off < length) ? a[idx + off - 1] : 0.0f;
    __syncthreads();
    bool j =true;
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        if(idx < BLOCK_SIZE){
            if(j){
                mem[BLOCK_SIZE + idx] =idx >= stride?mem[idx - stride] + mem[idx]:mem[idx];
            }else{
                mem[idx] = idx >= stride? mem[BLOCK_SIZE + idx -stride] + mem[BLOCK_SIZE + idx] : mem[idx + BLOCK_SIZE];
            }
            j=!j;
        }
        __syncthreads();
    }
    if (idx + off< length){
        int res = j?mem[idx]:mem[idx+32];
        b[idx + off] =  res + lite[blockIdx.x];
    }
        
}

void scan(int length){
    float *a=nullptr;
    float *b =nullptr;
    // float *lite = nullptr
    float *comp =nullptr;
    int block_num = (length + BLOCK_SIZE -1)/BLOCK_SIZE;
    cudaMallocHost(&a, sizeof(float) * length);
    cudaMallocHost(&b, sizeof(float) * length);
    // cudaMallocHost(&lite, sizeof(float) * block_num);
    cudaMallocHost(&comp, sizeof(float) * length);

    Init(a,length);
    Init(b,length,false);
    // Init(lite,block_num,false);
    Init(comp,length,false);

    float *dev_a=nullptr;
    float *dev_b=nullptr;
    float *dev_lite1=nullptr;
    float *dev_lite2=nullptr;

    cudaMalloc(&dev_a, sizeof(float) * length);
	cudaMalloc(&dev_b, sizeof(float) * length);
    cudaMalloc(&dev_lite1, sizeof(float) * block_num);
    cudaMalloc(&dev_lite2, sizeof(float) * block_num);

    cudaMemcpy(dev_a, a, length * sizeof(float), cudaMemcpyHostToDevice);
    
    reduce<<<block_num,BLOCK_SIZE>>>(dev_a,dev_lite1,length);
    exclusive_scan_lite<<<1,32>>>(dev_lite1,dev_lite2,block_num);
    exclusive_scan_block<<<block_num,BLOCK_SIZE>>>(dev_a,dev_lite2,dev_b,length);

    cudaMemcpy(b, dev_b, length * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());
	Hillis_Steele_cpu(a,comp,length);
	if(equal(b,comp,length)){
		std::cout<<"Functional SCAN Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional SCAN Testing WRONG!!!"<<std::endl;
	}

    for(int i=256;i<276;i++){
        std::cout<<a[i]<<" ";
    }
    std::cout<<std::endl;
    for(int i=256;i<276;i++){
        std::cout<<b[i]<<" ";
    }
    std::cout<<std::endl;
    for(int i=256;i<276;i++){
        std::cout<<comp[i]<<" ";
    }
	cudaFree(dev_a);
	cudaFree(dev_b);
    cudaFree(dev_lite1);
    cudaFree(dev_lite2);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(comp);
}

int main(){
    scan(511);
}
