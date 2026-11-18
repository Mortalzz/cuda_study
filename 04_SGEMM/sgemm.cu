#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include"../common/cuda_check.cuh"
#include"../common/reference.h"
#include"../common/timer.cuh"
#include"../common/benchmark.cuh"
#include<stdio.h>
#include<iostream>

#define TILE_Y 32
#define TILE_X 32
#define TILE_K 32
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


__device__ void load_a_float4(const float *a,float (*mema)[BK],int row_0,int tid,
                             int k_start,int m,int k,int nthreads)
{
    int total = BM * (BK / 4);
    for(int idx = tid;idx<total;idx+=nthreads){
        int r = idx / (BK /4);
        int c4 = (idx % (BK /4)) * 4;
        int gr = row_0 + r;
        int gc = c4 + k_start;
        float4 v;
        if(gr < m && gc + 3 <k){
            v = *reinterpret_cast<const float4*>(&a[gr * k + gc]);
        }else{
            v = make_float4(0, 0, 0, 0);    // 整段越界 → 全 0
        }
        *reinterpret_cast<float4*>(&mema[r][c4]) = v;
    }
}

__device__ void load_a_scalar(const float *a,float (*mema)[BK],int row_0,int tid,
                             int k_start,int m,int k,int nthreads)
{
    for(int idx =tid;idx < BM * BK;idx += nthreads){
        int r = idx / BK;
        int c = idx % BK;
        int gr = r + row_0;
        int gc = k_start + c;
        mema[r][c] = (gr< m && gc < k) ? a[gr * k + gc] : 0.0f;
    }
}

__device__ void load_b_float4(const float *b,float (*memb)[BN],int col_0,int tid,
                             int k_start,int k,int n,int nthreads)
{
    int total = BK * (BN / 4);
    for(int idx = tid;idx<total;idx+=nthreads){
        int r = idx / (BN /4);
        int c4 = (idx % (BN /4)) * 4;
        int gr = k_start + r;
        int gc = c4 + col_0;
        float4 v;
        if(gr < k && gc + 3 <n){
            v = *reinterpret_cast<const float4*>(&b[gr * n + gc]);
        }else{
            v = make_float4(0, 0, 0, 0);    // 整段越界 → 全 0
        }
        *reinterpret_cast<float4*>(&memb[r][c4]) = v;
    }
}

__device__ void load_b_scalar(const float *b,float (*memb)[BN],int col_0,int tid,
                             int k_start,int k,int n,int nthreads)
{
    for(int idx =tid;idx < BK * BN;idx += nthreads){
        int r = idx / BN;
        int c = idx % BN;
        int gr = r + k_start;
        int gc = col_0 + c;
        memb[r][c] = (gr< k && gc < n) ? b[gr * n + gc] : 0.0f;
    }
}


__global__ void gemm_reg_tiling_gpu(float *a,float *b,float *c,int m,int n,int k){
    __shared__ float mema[BM][BK];
    __shared__ float memb[BK][BN];
    float sum[REG_TILING][REG_TILING] = {0.0f};
    int row_0 = blockIdx.y * BM;
    int col_0 = blockIdx.x * BN;
    int tid =threadIdx.y * blockDim.x + threadIdx.x;
    bool a_aligned = (k % 4 == 0);
    bool b_aligned = (n % 4 == 0);
    for(int i =0;i<k;i+=BK){
        if(a_aligned){
            load_a_float4(a,mema,row_0,tid,i,m,k,TILE_Y *TILE_X);
        }else{
            load_a_scalar(a,mema,row_0,tid,i,m,k,TILE_Y *TILE_X);
        }

        if(b_aligned){
            load_b_float4(b,memb,col_0,tid,i,k,n,TILE_Y *TILE_X);
        }else{
            load_b_scalar(b,memb,col_0,tid,i,k,n,TILE_Y *TILE_X);
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
    //gemm_naive(m,n,k);
    //gemm_shared(m,n,k);
    gemm_reg_tiling(m,n,k);
}