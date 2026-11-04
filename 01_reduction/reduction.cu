#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"../common/cuda_check.cuh"
#include"../common/reference.h"
#include"../common/timer.cuh"
#include"../common/benchmark.cuh"
#include<stdio.h>
#include<iostream>
#define block_size 256

__global__ void naive_softmax_NHWC(float *a,float *b,int batch,int m,int n,int c){
    int off = blockIdx.x * blockDim.x * c + threadIdx.x * c;
    if(off >= batch * m * n * c){
        return;
    }
    float max =-FLT_MAX;
    float sum =0.0f;
    for(int i=0;i<c;i++){
        max = fmaxf(max,a[off + i]);
    }
    for(int i=0;i< c;i++){
        sum += __expf(a[off + i] - max);
    }
    float inv = 1.0f/sum;
    for(int i=0;i<c;i++){
        b[off + i] = __expf(a[off+i] - max) * inv;
    }
}

__global__ void shared_mem_softmax_NHWC(float *a,float *b,int batch,int m,int n,int c){
    __shared__ float mem[16][64];
    int row_off_block = blockIdx.x * 16;
    int row_off_thread = threadIdx.y;
    int row_off = (row_off_block + row_off_thread) * c ;
    int loop_times = (c+31)/32;
    mem[row_off_thread][threadIdx.x] = 0;
    mem[row_off_thread][threadIdx.x + 32] = -FLT_MAX;
    __syncthreads();
    for(int i=0;i<loop_times;i++){
        int col_off =  i * 32 + threadIdx.x;
        int off = row_off + col_off;
        if(col_off < c){
            float data = a[off];
            mem[row_off_thread][threadIdx.x] += __expf(data);
            mem[row_off_thread][threadIdx.x + 32] = fmaxf(data,mem[row_off_thread][threadIdx.x + 32]);
        }
    }
    __syncthreads();

    for(int i =16;i>0;i/=2){
        if(threadIdx.x < i){
            mem[row_off_thread][threadIdx.x] += mem[row_off_thread][threadIdx.x + i];
            mem[row_off_thread][threadIdx.x + 32] = fmaxf(mem[row_off_thread][threadIdx.x + 32],mem[row_off_thread][threadIdx.x + 32 + i]);
        }
        __syncthreads();
    }
    float max = mem[row_off_thread][32];
    float inv = 1.0f/(mem[row_off_thread][0] / __expf(max));
    for(int i=0;i<loop_times;i++){
        int col_off =  i * 32 + threadIdx.x;
        int off = row_off + col_off;
        if(col_off < c){
            b[off] = __expf(a[off] - max) * inv;
        }
    }
}

void naive_reduction_channel(int batch,int m, int n,int c){
    float *a = nullptr;
    float *b = nullptr;
    float *comp = nullptr;

    int sz = batch * m * n * c;

    cudaMallocHost(&a, sizeof(float) * sz);
    cudaMallocHost(&b, sizeof(float) * sz);
    cudaMallocHost(&comp, sizeof(float) * sz);

    Init(a,sz);
    Init(b,sz,false);
    Init(comp,sz,false);

    float *dev_a=nullptr;
    float *dev_b=nullptr;

    cudaMalloc(&dev_a,sizeof(float) * sz);
    cudaMalloc(&dev_b,sizeof(float) * sz);

    cudaMemcpy(dev_a,a,sizeof(float) * sz ,cudaMemcpyHostToDevice);
    dim3 block = dim3(256,1,1);
    dim3 grid = dim3((m * n * batch + 255)/256,1,1);
    // CudaTimer timer;
    // timer.start();
    naive_softmax_NHWC<<<grid,block>>>(dev_a,dev_b,batch,m,n,c);
    // timer.stop();
    cudaMemcpy(b,dev_b,sizeof(float) * sz ,cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
    softmax_cpu_NHWC(comp,a,batch,m,n,c);

    if(equal(b,comp,sz)){
		std::cout<<"Functional Naive_Softmax_NHWC Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional Naive_Softmax_NHWC Testing WRONG!!!"<<std::endl;
	}
    auto naive  = [&] { naive_softmax_NHWC<<<grid, block>>>(dev_a,dev_b,batch,m,n,c); };
    benchmark("naive" ,naive,3,10,2.0f * double(sz) * 4.0f);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(comp);
}

void shared_mem_reduction_channel(int batch,int m, int n,int c){
    float *a = nullptr;
    float *b = nullptr;
    float *comp = nullptr;

    int sz = batch * m * n * c;

    cudaMallocHost(&a, sizeof(float) * sz);
    cudaMallocHost(&b, sizeof(float) * sz);
    cudaMallocHost(&comp, sizeof(float) * sz);

    Init(a,sz);
    Init(b,sz,false);
    Init(comp,sz,false);

    float *dev_a=nullptr;
    float *dev_b=nullptr;

    cudaMalloc(&dev_a,sizeof(float) * sz);
    cudaMalloc(&dev_b,sizeof(float) * sz);

    cudaMemcpy(dev_a,a,sizeof(float) * sz ,cudaMemcpyHostToDevice);
    dim3 block = dim3(32,16,1);
    dim3 grid = dim3((m * n * batch + 15)/16,1,1);
    shared_mem_softmax_NHWC<<<grid,block>>>(dev_a,dev_b,batch,m,n,c);
    cudaMemcpy(b,dev_b,sizeof(float) * sz ,cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
    softmax_cpu_NHWC(comp,a,batch,m,n,c);

    if(equal(b,comp,sz)){
		std::cout<<"Functional shared_Softmax_NHWC Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional shared_Softmax_NHWC Testing WRONG!!!"<<std::endl;
	}
    auto shared  = [&] { shared_mem_softmax_NHWC<<<grid, block>>>(dev_a,dev_b,batch,m,n,c); };
    benchmark("shared" ,shared,3,10,2.0f * double(sz) * 4.0f);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(comp);
}

int main(){
    //naive_reduction_channel(32,57,56,256);
    shared_mem_reduction_channel(32,57,56,256);
}