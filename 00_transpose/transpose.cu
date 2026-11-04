#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"../common/cuda_check.cuh"
#include"../common/timer.cuh"
#include"../common/reference.h"
#include<stdio.h>
#include<iostream>

#define INDX( row, col, ld ) ( ( (row) * (ld) ) + (col) )


__global__ void transpose_without_shared(int* c, int* a,int m,int n) {
	int myrow = blockIdx.y * blockDim.y + threadIdx.y;
	int mycol = blockIdx.x * blockDim.x + threadIdx.x;

	if (myrow < m && mycol < n) {
		c[INDX(mycol, myrow, m)] = a[INDX(myrow, mycol, n)];
	}
	return;	
}

__global__ void transpose_with_shared(int* c, int* a, int m, int n) {
	int myrow = blockIdx.y * blockDim.y + threadIdx.y;
	int mycol = blockIdx.x * blockDim.x + threadIdx.x;

	int Ctile_y = blockIdx.x * blockDim.x;
	int Ctile_x = blockIdx.y * blockDim.y;
	__shared__ int mem[16][17];

	if (myrow < m && mycol < n) {
		mem[threadIdx.x][threadIdx.y] = a[INDX(myrow, mycol, n)];
	}

	__syncthreads();

	int c_row = Ctile_y + threadIdx.y;
	int c_col = Ctile_x + threadIdx.x;

	if (c_row < n && c_col < m) {
		c[INDX(c_row, c_col, m)] = mem[threadIdx.y][threadIdx.x];
	}
	return;
}


void transpose(int m,int n) {
	int* a = nullptr;
	int* b = nullptr;
	int* c = nullptr;
	int* comp_c = nullptr;
	int *dev_a = nullptr;
	int *dev_b = nullptr;
	int *dev_c = nullptr;

	int sz = m * n;
	cudaMallocHost(&a, sizeof(int) * sz);
	cudaMallocHost(&b, sizeof(int) * sz);
	cudaMallocHost(&c, sizeof(int) * sz);
	cudaMallocHost(&comp_c,sizeof(int) * sz);
	cudaMalloc(&dev_a, sizeof(int) * sz);
	cudaMalloc(&dev_b, sizeof(int) * sz);
	cudaMalloc(&dev_c, sizeof(int) * sz);

	Init(a, sz);
	Init(c, sz,false);
	dim3 block = dim3(16, 16);
	dim3 grid = dim3((n + 15) / 16, (m + 15) / 16);

	cudaMemcpy(dev_a, a, sz * sizeof(int), cudaMemcpyHostToDevice);

	CudaTimer timer;
	timer.start();
	transpose_without_shared <<<grid, block >>> (dev_c, dev_a, m,n);;
	timer.stop();
	double ms = timer.elapsed_ms();
	cudaMemcpy(c, dev_c, sz * sizeof(int), cudaMemcpyDeviceToHost);
	CUDA_CHECK(cudaDeviceSynchronize());
	transpose_cpu(comp_c,a,m,n);
	if(equal(c,comp_c,sz)){
		std::cout<<"Functional Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional Testing WRONG!!!"<<std::endl;
	}

	double gbs = 2 * double(sz) * 4 / ms / 1000000.0;
	std::cout<<"Final Speed:"<<std::endl;
	std::cout<<gbs<<"GB/S";

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
	cudaFreeHost(comp_c);
}



int main() {
	transpose(10240,10240);
	return 0;
}
