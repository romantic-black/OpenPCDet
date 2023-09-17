/*
Stacked-batch-data version of ball query, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_kernel_stack(int B, int M, float radius, int nsample, \
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    // 获取全局索引
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    // 要获取当前点所在 batch 及 batch 结束点位置 (new_xyz)
    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;    // 获取 batch 的起始点位置 (xyz)
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];
    // for (int k = 0; k < bs_idx; k++) new_xyz_batch_start_idx += new_xyz_batch_cnt[k];

    // 移动指针
    new_xyz += pt_idx * 3;              // 质心位置
    xyz += xyz_batch_start_idx * 3;     // 体素在该 batch 下的起始点
    idx += pt_idx * nsample;            // 输出数组

    float radius2 = radius * radius;    //
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];      // 该 batch 中有几个点

    int cnt = 0;
    for (int k = 0; k < n; ++k) {       // 遍历 batch 中所有体素，居然是从头到尾遍历，不过好像也只能这样
        float x = xyz[k * 3 + 0];       // 体素位置
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        // 体素到质心的距离
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            // 如果是第一个在半径内找到的点，将其索引值填充到整个idx数组中
            // 这样做是为了确保至少有一个有效的索引值，即使没有足够的点满足条件
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;   // 也就是说，并没有按论文那样，直接用曼哈顿距离找？而是用 O(n) 的方式
            ++cnt;          // 不过确实，附近的点不一定有值，他必须选有值的
            if (cnt >= nsample) break;
        }
    }
    if (cnt == 0) idx[0] = -1;  // 如果没找到点则全设为 -1
}

// 内核启动函数，看来每个 cu 必须有这个东西，因为调用核函数的函数也需要使用 cu 的语法
void ball_query_kernel_launcher_stack(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    cudaError_t err;
    // dim3 变量: 被传递给核函数，用于配置并行参数
    // DIVUP: 向上整除，确保有足够 block 处理 M 个数据点
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);    // 每个 block 中包含多少 threads，预定义为 256

    ball_query_kernel_stack<<<blocks, threads>>>(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
