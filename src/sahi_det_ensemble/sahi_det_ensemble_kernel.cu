/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * SAHI Detection Ensemble CUDA Kernels
 *
 * 三个核心操作，全部在 GPU 上完成：
 *   1. filter_and_offset — 置信度过滤 + 偏移校正 + 裁剪（一体化）
 *   2. nms_per_class     — 逐类 NMS
 *   3. topk_pad          — Top-K 选择并填充到固定输出
 */

#include "sahi_det_ensemble/sahi_det_ensemble_kernel.hpp"
#include <cuda_runtime.h>

namespace sahi_det_ensemble
{

// ====================================================================
//  1. filter_and_offset: 置信度过滤 + 偏移校正 + 裁剪（一体化）
// ====================================================================

static __global__ void filter_offset_kernel(
    const int *__restrict__ det_num_dets,
    const float *__restrict__ det_boxes,
    const float *__restrict__ det_scores,
    const int *__restrict__ det_classes,
    const int *__restrict__ slice_offsets,
    int num_slices, int max_dets, int box_dim,
    float conf_threshold, int img_w, int img_h,
    float *__restrict__ out_boxes,
    float *__restrict__ out_scores,
    int *__restrict__ out_classes,
    int *__restrict__ out_slice_idx,
    int *__restrict__ d_out_count,
    int max_output)
{
    __shared__ int s_base, s_prefix[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_slices * max_dets;
    bool valid = false;
    float score = 0.0f;

    if (tid < total) {
        int si = tid / max_dets;
        int di = tid % max_dets;
        if (di < det_num_dets[si]) {
            score = det_scores[tid];
            if (score >= conf_threshold) valid = true;
        }
    }

    // 线程内 prefix-sum 计数（确定性的 block 内编号）
    int v = valid ? 1 : 0;
    s_prefix[threadIdx.x] = v;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {
        int add = (threadIdx.x >= off) ? s_prefix[threadIdx.x - off] : 0;
        __syncthreads();
        s_prefix[threadIdx.x] += add;
        __syncthreads();
    }

    int block_cnt = s_prefix[blockDim.x - 1];
    int my_pos = valid ? (s_prefix[threadIdx.x] - 1) : -1;

    // 每 block 仅一次 atomicAdd 预留全局区间
    if (threadIdx.x == 0 && block_cnt > 0)
        s_base = atomicAdd(d_out_count, block_cnt);
    __syncthreads();

    int gidx = s_base + my_pos;
    if (gidx < 0 || gidx >= max_output || !valid) return;

    int si = tid / max_dets;
    int ox = slice_offsets[si * 4 + 0];
    int oy = slice_offsets[si * 4 + 1];
    float fw = static_cast<float>(img_w);
    float fh = static_cast<float>(img_h);

    if (box_dim == 5) {
        out_boxes[gidx * 5 + 0] = fminf(fmaxf(det_boxes[tid * 5 + 0] + ox, 0.0f), fw);
        out_boxes[gidx * 5 + 1] = fminf(fmaxf(det_boxes[tid * 5 + 1] + oy, 0.0f), fh);
        out_boxes[gidx * 5 + 2] = det_boxes[tid * 5 + 2];
        out_boxes[gidx * 5 + 3] = det_boxes[tid * 5 + 3];
        out_boxes[gidx * 5 + 4] = det_boxes[tid * 5 + 4];
    } else {
        out_boxes[gidx * 4 + 0] = fminf(fmaxf(det_boxes[tid * 4 + 0] + ox, 0.0f), fw);
        out_boxes[gidx * 4 + 1] = fminf(fmaxf(det_boxes[tid * 4 + 1] + oy, 0.0f), fh);
        out_boxes[gidx * 4 + 2] = fminf(fmaxf(det_boxes[tid * 4 + 2] + ox, 0.0f), fw);
        out_boxes[gidx * 4 + 3] = fminf(fmaxf(det_boxes[tid * 4 + 3] + oy, 0.0f), fh);
    }
    out_scores[gidx] = score;
    out_classes[gidx] = det_classes[tid];
    out_slice_idx[gidx] = tid;
}

void filter_and_offset(
    const int *det_num_dets, const float *det_boxes,
    const float *det_scores, const int *det_classes,
    const int *slice_offsets,
    int num_slices, int max_dets, int box_dim,
    float conf_threshold, int img_w, int img_h,
    float *out_boxes, float *out_scores, int *out_classes,
    int *out_slice_idx, int *d_out_count,
    int max_output, cudaStream_t stream)
{
    cudaMemsetAsync(d_out_count, 0, sizeof(int), stream);
    int total = num_slices * max_dets;
    int block = 256;
    int grid = (total + block - 1) / block;
    filter_offset_kernel<<<grid, block, 0, stream>>>(
        det_num_dets, det_boxes, det_scores, det_classes,
        slice_offsets, num_slices, max_dets, box_dim,
        conf_threshold, img_w, img_h,
        out_boxes, out_scores, out_classes,
        out_slice_idx, d_out_count, max_output);
}

// ====================================================================
//  1b. strided_copy: 将检测器 [n, actual_dets, 4/1] 拷贝到 [n, max_dets, 4/1]
// ====================================================================

static __global__ void strided_copy_boxes_kernel(
    const float *__restrict__ src, float *__restrict__ dst,
    int n, int src_stride, int dst_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * src_stride;
    if (idx >= total) return;
    int img = idx / src_stride;
    int off = idx % src_stride;
    dst[img * dst_stride + off] = src[idx];
}

void strided_copy_boxes(
    const float *src, float *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream)
{
    int total = n * src_stride;
    int block = 256;
    int grid = (total + block - 1) / block;
    strided_copy_boxes_kernel<<<grid, block, 0, stream>>>(
        src, dst, n, src_stride, dst_stride);
}

static __global__ void strided_copy_scores_kernel(
    const float *__restrict__ src, float *__restrict__ dst,
    int n, int src_stride, int dst_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * src_stride;
    if (idx >= total) return;
    int img = idx / src_stride;
    int off = idx % src_stride;
    dst[img * dst_stride + off] = src[idx];
}

void strided_copy_scores(
    const float *src, float *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream)
{
    int total = n * src_stride;
    int block = 256;
    int grid = (total + block - 1) / block;
    strided_copy_scores_kernel<<<grid, block, 0, stream>>>(
        src, dst, n, src_stride, dst_stride);
}

static __global__ void strided_copy_classes_kernel(
    const int *__restrict__ src, int *__restrict__ dst,
    int n, int src_stride, int dst_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * src_stride;
    if (idx >= total) return;
    int img = idx / src_stride;
    int off = idx % src_stride;
    dst[img * dst_stride + off] = src[idx];
}

void strided_copy_classes(
    const int *src, int *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream)
{
    int total = n * src_stride;
    int block = 256;
    int grid = (total + block - 1) / block;
    strided_copy_classes_kernel<<<grid, block, 0, stream>>>(
        src, dst, n, src_stride, dst_stride);
}

static __global__ void strided_copy_keypoints_kernel(
    const float *__restrict__ src, float *__restrict__ dst,
    int n, int src_stride, int dst_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * src_stride;
    if (idx >= total) return;
    int img = idx / src_stride;
    int off = idx % src_stride;
    dst[img * dst_stride + off] = src[idx];
}

void strided_copy_keypoints(
    const float *src, float *dst,
    int n, int src_stride, int dst_stride,
    cudaStream_t stream)
{
    int total = n * src_stride;
    int block = 256;
    int grid = (total + block - 1) / block;
    strided_copy_keypoints_kernel<<<grid, block, 0, stream>>>(
        src, dst, n, src_stride, dst_stride);
}

// ====================================================================
//  2. nms_per_class: 确定性逐类 NMS（GPU 版），支持 AABB 与 OBB
// ====================================================================

#include "common/iou.cuh"
using common_iou::box_iou;
using common_iou::box_probiou;

static __global__ void fast_nms_kernel(
    const float *__restrict__ boxes,
    const float *__restrict__ scores,
    const int   *__restrict__ classes,
    int N, int box_dim, float iou_threshold,
    int *__restrict__ keep_flags)   // 1=keep, 0=suppressed
{
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= N) return;

    // 确定性排序：score 降序，index 升序
    float cur_score = scores[pos];
    int   cur_class = classes[pos];

    for (int i = 0; i < N; ++i) {
        if (i == pos || classes[i] != cur_class) continue;

        float item_score = scores[i];
        if (item_score > cur_score || (item_score == cur_score && i < pos)) {
            float iou;
            if (box_dim == 5) {
                iou = box_probiou(
                    boxes[pos*5+0], boxes[pos*5+1], boxes[pos*5+2], boxes[pos*5+3], boxes[pos*5+4],
                    boxes[i*5+0],   boxes[i*5+1],   boxes[i*5+2],   boxes[i*5+3],   boxes[i*5+4]);
            } else {
                iou = box_iou(
                    boxes[pos*4+0], boxes[pos*4+1], boxes[pos*4+2], boxes[pos*4+3],
                    boxes[i*4+0],   boxes[i*4+1],   boxes[i*4+2],   boxes[i*4+3]);
            }
            if (iou > iou_threshold) {
                keep_flags[pos] = 0;
                return;
            }
        }
    }
}

static __global__ void compact_kernel(
    const int *__restrict__ sorted_idx,
    const int *__restrict__ flags, int N,
    int *__restrict__ keep, int *__restrict__ d_num)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (flags[i]) {
        int pos = atomicAdd(d_num, 1);
        keep[pos] = sorted_idx[i];
    }
}

static __global__ void fill_identity_kernel(int *arr, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) arr[i] = i;
}

void nms_per_class(
    const float *boxes, const float *scores, const int *classes,
    int N, float iou_threshold, int num_classes, int box_dim,
    int *keep, int *d_num_kept,
    int *d_class_offsets, int *d_counters,
    int *d_flags, cudaStream_t stream)
{
    if (N <= 0) { cudaMemsetAsync(d_num_kept, 0, sizeof(int), stream); return; }

    int block = 256;
    int grid = (N + block - 1) / block;

    // d_flags 初始化为 1（全部 keep）
    cudaMemsetAsync(d_flags, 1, N * sizeof(int), stream);

    // 确定性 NMS：score 降序 + index 升序 tiebreaker
    fast_nms_kernel<<<grid, block, 0, stream>>>(
        boxes, scores, classes, N, box_dim, iou_threshold, d_flags);

    // 初始化 keep 为 [0,1,2,...,N-1]（compact_kernel 需要读取）
    fill_identity_kernel<<<grid, block, 0, stream>>>(keep, N);

    // compact: 将 keep_flags==1 的索引收集到 keep 数组
    cudaMemsetAsync(d_num_kept, 0, sizeof(int), stream);
    compact_kernel<<<grid, block, 0, stream>>>(keep, d_flags, N, keep, d_num_kept);
}

// ====================================================================
//  3. topk_pad: Top-K 选择 + 填充到固定维度
// ====================================================================

static __global__ void topk_pad_kernel(
    const float *__restrict__ boxes,
    const float *__restrict__ scores,
    const int *__restrict__ classes,
    int N, int max_dets,
    float *__restrict__ out_boxes,
    float *__restrict__ out_scores,
    int *__restrict__ out_classes,
    int *__restrict__ d_num_dets)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_dets) return;

    if (i < N) {
        out_scores[i] = scores[i];
        out_boxes[i * 4 + 0] = boxes[i * 4 + 0];
        out_boxes[i * 4 + 1] = boxes[i * 4 + 1];
        out_boxes[i * 4 + 2] = boxes[i * 4 + 2];
        out_boxes[i * 4 + 3] = boxes[i * 4 + 3];
        out_classes[i] = classes[i];
    } else {
        out_scores[i] = 0.0f;
        out_boxes[i * 4 + 0] = 0.0f;
        out_boxes[i * 4 + 1] = 0.0f;
        out_boxes[i * 4 + 2] = 0.0f;
        out_boxes[i * 4 + 3] = 0.0f;
        out_classes[i] = -1;
    }
    if (i == 0) *d_num_dets = min(N, max_dets);
}

void topk_pad(
    const float *boxes, const float *scores, const int *classes,
    int N, int max_detections,
    float *out_boxes, float *out_scores, int *out_classes,
    int *d_num_dets, cudaStream_t stream)
{
    int block = 256;
    int grid = (max_detections + block - 1) / block;
    topk_pad_kernel<<<grid, block, 0, stream>>>(
        boxes, scores, classes, N, max_detections,
        out_boxes, out_scores, out_classes, d_num_dets);
}

} // namespace sahi_det_ensemble
