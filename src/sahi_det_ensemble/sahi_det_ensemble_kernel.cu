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
    int num_slices, int max_dets,
    float conf_threshold, int img_w, int img_h,
    float *__restrict__ out_boxes,
    float *__restrict__ out_scores,
    int *__restrict__ out_classes,
    int *__restrict__ out_slice_idx,
    int *__restrict__ d_out_count,
    int max_output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_slices * max_dets;
    if (tid >= total) return;

    int si = tid / max_dets;
    int di = tid % max_dets;
    if (di >= det_num_dets[si]) return;

    float score = det_scores[tid];
    if (score < conf_threshold) return;

    int ox = slice_offsets[si * 4 + 0];
    int oy = slice_offsets[si * 4 + 1];
    float fw = static_cast<float>(img_w);
    float fh = static_cast<float>(img_h);

    float x1 = fminf(fmaxf(det_boxes[tid * 4 + 0] + ox, 0.0f), fw);
    float y1 = fminf(fmaxf(det_boxes[tid * 4 + 1] + oy, 0.0f), fh);
    float x2 = fminf(fmaxf(det_boxes[tid * 4 + 2] + ox, 0.0f), fw);
    float y2 = fminf(fmaxf(det_boxes[tid * 4 + 3] + oy, 0.0f), fh);

    int idx = atomicAdd(d_out_count, 1);
    if (idx >= max_output) return;

    out_boxes[idx * 4 + 0] = x1;
    out_boxes[idx * 4 + 1] = y1;
    out_boxes[idx * 4 + 2] = x2;
    out_boxes[idx * 4 + 3] = y2;
    out_scores[idx] = score;
    out_classes[idx] = det_classes[tid];
    out_slice_idx[idx] = tid;  // 存储原始 flat index（可还原 si 和 di）
}

void filter_and_offset(
    const int *det_num_dets, const float *det_boxes,
    const float *det_scores, const int *det_classes,
    const int *slice_offsets,
    int num_slices, int max_dets,
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
        slice_offsets, num_slices, max_dets,
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
//  2. nms_per_class: 逐类 NMS（GPU 版）
// ====================================================================

static __global__ void class_offset_kernel(
    const int *__restrict__ classes, int N,
    int *__restrict__ offsets, int num_classes)
{
    for (int i = threadIdx.x; i <= num_classes; i += blockDim.x) offsets[i] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        atomicAdd(&offsets[classes[i]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int c = 0; c < num_classes; ++c) {
            int cnt = offsets[c];
            offsets[c] = sum;
            sum += cnt;
        }
        offsets[num_classes] = sum;
    }
}

static __global__ void scatter_kernel(
    const float *__restrict__ scores,
    const int *__restrict__ classes, int N,
    int *__restrict__ out_indices,
    const int *__restrict__ offsets,
    int *__restrict__ counters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int c = classes[idx];
    int pos = atomicAdd(&counters[c], 1);
    out_indices[offsets[c] + pos] = idx;
}

static __global__ void nms_kernel(
    const float *__restrict__ boxes,
    const int *__restrict__ sorted_idx,
    const int *__restrict__ offsets,
    int num_classes, float iou_threshold,
    int *__restrict__ flags)
{
    int c = blockIdx.x;
    int start = offsets[c];
    int cnt = offsets[c + 1] - start;
    if (cnt <= 0) return;

    int tid = threadIdx.x;
    extern __shared__ float smem[];
    float *sx1 = smem;
    float *sy1 = smem + cnt;
    float *sx2 = smem + cnt * 2;
    float *sy2 = smem + cnt * 3;
    float *sarea = smem + cnt * 4;

    if (tid < cnt) {
        int bi = sorted_idx[start + tid];
        sx1[tid] = boxes[bi * 4 + 0];
        sy1[tid] = boxes[bi * 4 + 1];
        sx2[tid] = boxes[bi * 4 + 2];
        sy2[tid] = boxes[bi * 4 + 3];
        sarea[tid] = (sx2[tid] - sx1[tid]) * (sy2[tid] - sy1[tid]);
    }
    __syncthreads();

    for (int i = tid; i < cnt; i += blockDim.x) {
        bool keep = true;
        for (int j = 0; j < i; ++j) {
            if (!flags[start + j]) continue;
            float inter = fmaxf(0.0f, fminf(sx2[i], sx2[j]) - fmaxf(sx1[i], sx1[j]))
                        * fmaxf(0.0f, fminf(sy2[i], sy2[j]) - fmaxf(sy1[i], sy1[j]));
            float iou = inter / (sarea[i] + sarea[j] - inter + 1e-8f);
            if (iou > iou_threshold) { keep = false; break; }
        }
        flags[start + i] = keep ? 1 : 0;
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

void nms_per_class(
    const float *boxes, const float *scores, const int *classes,
    int N, float iou_threshold, int num_classes,
    int *keep, int *d_num_kept,
    int *d_class_offsets, int *d_counters,
    int *d_flags, cudaStream_t stream)
{
    if (N <= 0) { cudaMemsetAsync(d_num_kept, 0, sizeof(int), stream); return; }

    const int BS = 256;

    class_offset_kernel<<<1, BS, 0, stream>>>(classes, N, d_class_offsets, num_classes);

    cudaMemsetAsync(d_counters, 0, num_classes * sizeof(int), stream);
    int grid = (N + BS - 1) / BS;
    scatter_kernel<<<grid, BS, 0, stream>>>(scores, classes, N, keep, d_class_offsets, d_counters);

    cudaMemsetAsync(d_flags, 1, N * sizeof(int), stream);

    int max_cls = min(N, 1024);
    size_t smem_sz = static_cast<size_t>(max_cls) * 5 * sizeof(float);
    nms_kernel<<<num_classes, max_cls, smem_sz, stream>>>(
        boxes, keep, d_class_offsets, num_classes, iou_threshold, d_flags);

    cudaMemsetAsync(d_num_kept, 0, sizeof(int), stream);
    compact_kernel<<<grid, BS, 0, stream>>>(keep, d_flags, N, keep, d_num_kept);
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
