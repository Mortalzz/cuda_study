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

void stencil_cpu(float *a,float *b,int m,int n){
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum =0.0f;
            int cnt_mn = i*n+j;
            for(int k=-1;k<=1;k++){
                for(int l=-1;l<=1;l++){
                    int x = i+k;
                    int y = j+l;
                    if(x>=0 && x<m && y>=0 && y<n){
                        sum+=a[x*n+y];
                    }
                }
            }
            b[cnt_mn] = sum/9.0f;
        }
    }
}