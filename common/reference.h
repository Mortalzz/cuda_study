#include<iostream>
#include<random>
#include<float.h>
#include <cstdlib>


void Init(float *a, int size,bool judge =true) {
    std::mt19937 gen(42);                                  // 换成任何固定整数，比如 0、1234、2024
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < size;i++){
        a[i] = judge?dist(gen) * 10.0:0;
    }
}

void Init(int *a, int size,bool judge =true) {
    std::mt19937 gen(42);                                  // 换成任何固定整数，比如 0、1234、2024
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < size;i++){
        a[i] = judge?static_cast<int>(dist(gen) * 10):0;
    }
}



bool equal(int *a,int *b,int size){
    bool result = true;
    for(int i=0;i<size;i++){
        if(a[i]!=b[i]){
            result = false;
            break;
        }
    }
    return result;
}

bool equal(float *a,float *b,int size){
    bool result = true;
    for(int i=0;i<size;i++){
        if(fabs(a[i] - b[i]) > 1e-3){
            result = false;
            break;
        }
    }
    return result;
}



void transpose_cpu(int *c, int* a,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            c[j * m + i] = a[i * n + j];
        }
    }
}


void softmax_cpu_NHWC(float *b,float *a,int batch,int m,int n,int c){
    for(int i=0;i<batch * m * n;i++){
        float sum =0.0f;
        float max = -FLT_MAX;
        int cnt_mn =i * c;
        for(int j =0;j<c;j++){
            max = fmaxf(max,a[cnt_mn + j]);
        }
        for(int j=0;j<c;j++){
            sum+= expf(a[cnt_mn + j] - max);
        }
        float inv =1.0/sum;
        for(int j=0;j<c;j++){
            b[cnt_mn + j] = expf(a[cnt_mn + j]-max) *inv;
        }
    }
}


void Hillis_Steele_cpu(float *a,float *b,int size){
    for(int i =0 ;i<size;i++){
        b[i] = i==0? 0:a[i-1]+b[i-1];
    }
}

void stencil_clamp_cpu(float *a,float *b,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            int yt = i>0 ? (i-1):0;
            int yb = i< (m-1)?(i+1):m-1;
            int xl = j>0?(j-1):0;
            int xr = j<(j-1)? (j+1):n-1;
            b[i*n+j] = 0.5 * a[i * n +j] 
                    + 0.125 * a[yt * n + j]
                    + 0.125 * a[yb * n + j]
                    + 0.125 * a[i * n + xl];
                    + 0.125 * a[i * n + xr]
        }
    }
}