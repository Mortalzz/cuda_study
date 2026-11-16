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
#define TILE_K 16
#define REG_TILING 4
#define BM (TILE_Y * REG_TILING)   // block 覆盖的 C 行数
#define BN (TILE_X * REG_TILING)   // block 覆盖的 C 列数
#define BK TILE_K                  // K 方向 tile


__global__ void gemm_naive_gpu(float *a,float *b,float *c,int m,int n,int k){
    int row_off = blockIdx.y *TILE_Y + threadIdx.y;
    int col_off = blockIdx.x *TILE_X + threadIdx.x;

    int idx = row_off * n + col_off;
    float sum = 0.0f;
    if(row_off <m && col_off<n){
        for(int i =0;i< k;i++){
            sum += a[row_off * k + i] * b[i * n + col_off];
        }
        c[idx] = sum;
    }
}

__global__ void gemm_shared_gpu(float *a,float *b,float *c,int m,int n,int k){
    __shared__ float mema[TILE_Y][TILE_X];
    __shared__ float memb[TILE_Y][TILE_X];
    int row_block = blockIdx.y * blockDim.y;
    int col_block = blockIdx.x * blockDim.x;
    int row_thread = threadIdx.y;
    int col_thread = threadIdx.x;
    float sum = 0.0f;
    int row = row_block + row_thread;
    int col = col_block + col_thread;
    for(int i =0;i<k;i+=TILE_X){
        mema[row_thread][col_thread] = (row < m && col_thread + i < k) ? a[row * k + col_thread + i] : 0.0f;
        memb[row_thread][col_thread] = (row_thread + i < k && col < n) ? b[(row_thread + i) * n + col] : 0.0f;
        __syncthreads();

        for(int j =0;j<TILE_X;j++){
            sum += mema[threadIdx.y][j] * memb[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(row< m&&col<n){
        c[row * n + col] = sum;
    }
}

__global__ void gemm_reg_tiling_gpu(float *a,float *b,float *c,int m,int n,int k){
    __shared__ float mema[BM][BK];
    __shared__ float memb[BK][BN];
    float sum[REG_TILING][REG_TILING] = {0.0f};
    int row_0 = blockIdx.y * BM;
    int col_0 = blockIdx.x * BN;
    int tid =threadIdx.y * blockDim.x + threadIdx.x;
    for(int i =0;i<k;i+=BK){
        for(int j = tid;j<BM * BK;j+=TILE_X *TILE_Y){
            int r = j/BK;
            int c = j%BK;
            int gr = row_0 + r;
            int gc = c + i;
            mema[r][c] = (gr< m && gc < k) ? a[gr * k + gc] : 0.0f;
        }
        for(int j = tid;j<BK * BN;j+=TILE_X *TILE_Y){
            int r = j/BN;
            int c = j%BN;
            int gr = i + r;
            int gc = col_0 + c;
            memb[r][c] = (gr< k && gc < n) ? b[gr * n + gc] : 0.0f;
        }
        __syncthreads();

        for(int kk=0;kk<BK;kk++){
            float av[REG_TILING],bv[REG_TILING];
            for(int ii=0;ii<REG_TILING;ii++){
                av[ii] = mema[threadIdx.y * REG_TILING + ii][kk];
            }
            for(int jj=0;jj<REG_TILING;jj++){
                bv[jj] = memb[kk][threadIdx.x * REG_TILING +jj];
            }
            for(int ii=0;ii<REG_TILING;ii++){
                for(int jj=0;jj<REG_TILING;jj++){
                    sum[ii][jj] +=av[ii] * bv[jj];
                }
            }
        }
        __syncthreads();
    }
    for(int i =0;i<REG_TILING;i++){
        int gr = row_0 + threadIdx.y * REG_TILING + i;
        if (gr >= m) continue;
        for (int j = 0; j < REG_TILING; j++) {
            int gc = col_0 + threadIdx.x * REG_TILING + j;
            if (gc >= n) continue;
            c[gr * n + gc] = sum[i][j];
        }
    }
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
    cudaMalloc(&dev_a,sizeof(float) * mk);
    cudaMalloc(&dev_b,sizeof(float) * nk);
    cudaMalloc(&dev_c,sizeof(float) * mn);
    cudaMemcpy(dev_a,a,sizeof(float) * mk,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,sizeof(float) * nk,cudaMemcpyHostToDevice);
    dim3 grid = dim3((n + TILE_X - 1) / TILE_X,  (m + TILE_Y - 1) / TILE_Y); 
    dim3 block = dim3(TILE_X,TILE_Y);
    gemm_naive_gpu<<<grid,block>>>(dev_a,dev_b,dev_c,m,n,k);
    cudaMemcpy(c,dev_c,sizeof(float) * mn,cudaMemcpyDeviceToHost);
    gemm_cpu(a,b,comp,m,n,k);

    if(equal(c,comp,m * n)){
		std::cout<<"Functional GEMM_NAIVE Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional GEMM_NAIVE Testing WRONG!!!"<<std::endl;
	}
    auto sgemm_naive = [&] { gemm_naive_gpu<<<grid,block>>>(dev_a,dev_b,dev_c,m,n,k); };
    double bytes = (double)(m*k + k*n + m*n) * 4;
    double flops = 2.0 * m * n * k;

    benchmark("sgemm_naive" ,sgemm_naive,3,10, bytes,flops);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(comp);
}


void gemm_shared(int m,int n,int k){
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
    cudaMalloc(&dev_a,sizeof(float) * mk);
    cudaMalloc(&dev_b,sizeof(float) * nk);
    cudaMalloc(&dev_c,sizeof(float) * mn);
    cudaMemcpy(dev_a,a,sizeof(float) * mk,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,sizeof(float) * nk,cudaMemcpyHostToDevice);
    dim3 grid = dim3((n + TILE_X - 1) / TILE_X,  (m + TILE_Y - 1) / TILE_Y); 
    dim3 block = dim3(TILE_X,TILE_Y);
    gemm_shared_gpu<<<grid,block>>>(dev_a,dev_b,dev_c,m,n,k);
    cudaMemcpy(c,dev_c,sizeof(float) * mn,cudaMemcpyDeviceToHost);
    gemm_cpu(a,b,comp,m,n,k);
    if(equal(c,comp,m * n)){
		std::cout<<"Functional GEMM_SHARED Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional GEMM_SHARED Testing WRONG!!!"<<std::endl;
	}
    auto sgemm_shared = [&] { gemm_shared_gpu<<<grid,block>>>(dev_a,dev_b,dev_c,m,n,k); };
    double bytes = (double)(m*k + k*n + m*n) * 4;
    double flops = 2.0 * m * n * k;

    benchmark("sgemm_shared" ,sgemm_shared,3,10, bytes,flops);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(comp);
}


void gemm_reg_tiling(int m,int n,int k){
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
    cudaMalloc(&dev_a,sizeof(float) * mk);
    cudaMalloc(&dev_b,sizeof(float) * nk);
    cudaMalloc(&dev_c,sizeof(float) * mn);
    cudaMemcpy(dev_a,a,sizeof(float) * mk,cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b,b,sizeof(float) * nk,cudaMemcpyHostToDevice);
    dim3 grid = dim3((n + TILE_X * REG_TILING - 1) / (TILE_X * REG_TILING),  (m + TILE_Y * REG_TILING - 1) / (TILE_Y * REG_TILING)); 
    dim3 block = dim3(TILE_X,TILE_Y);
    gemm_reg_tiling_gpu<<<grid,block>>>(dev_a,dev_b,dev_c,m,n,k);
    cudaMemcpy(c,dev_c,sizeof(float) * mn,cudaMemcpyDeviceToHost);
    gemm_cpu(a,b,comp,m,n,k);
    if(equal(c,comp,m * n)){
		std::cout<<"Functional GEMM_REG_TILING Testing PASS!!!"<<std::endl;
	}else{
		std::cout<<"Functional GEMM_REG_TILING Testing WRONG!!!"<<std::endl;
	}
    auto sgemm_reg_tiling = [&] { gemm_reg_tiling_gpu<<<grid,block>>>(dev_a,dev_b,dev_c,m,n,k); };
    double bytes = (double)(m*k + k*n + m*n) * 4;
    double flops = 2.0 * m * n * k;

    benchmark("sgemm_reg_tiling" ,sgemm_reg_tiling,3,10, bytes,flops);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(comp);
}

int main(){
    int m = 1024;
    int n =2048;
    int k = 1024;
    gemm_naive(m,n,k);
    gemm_shared(m,n,k);
    gemm_reg_tiling(m,n,k);
}