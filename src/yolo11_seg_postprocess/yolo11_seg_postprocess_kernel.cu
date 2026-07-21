/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_seg_postprocess/yolo11_seg_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <float.h>

#include <cub/cub.cuh>

namespace yolo11_seg_postprocess
{

constexpr int kMaskOutputSize = 160;

static __device__ __forceinline__ float iou(
    float x1, float y1, float x2, float y2,
    float x1_, float y1_, float x2_, float y2_)
{
    float ix1 = fmaxf(x1, x1_);
    float iy1 = fmaxf(y1, y1_);
    float ix2 = fminf(x2, x2_);
    float iy2 = fminf(y2, y2_);

    float inter_w = fmaxf(0.0f, ix2 - ix1);
    float inter_h = fmaxf(0.0f, iy2 - iy1);
    float inter   = inter_w * inter_h;

    float area1 = (x2 - x1) * (y2 - y1);
    float area2 = (x2_ - x1_) * (y2_ - y1_);
    float uni   = area1 + area2 - inter + 1e-6f;

    return inter / uni;
}

template <typename T>
static __device__ __forceinline__ float read_output0(
    const T *input,
    int batch_idx,
    int anchor_idx,
    int channel_idx,
    int num_anchors,
    int num_channels,
    bool anchors_first)
{
    int idx;
    if (anchors_first)
    {
        idx = batch_idx * num_anchors * num_channels +
              anchor_idx * num_channels +
              channel_idx;
    }
    else
    {
        idx = batch_idx * num_channels * num_anchors +
              channel_idx * num_anchors +
              anchor_idx;
    }
    return static_cast<float>(input[idx]);
}

template <typename T>
static __device__ __forceinline__ float read_proto(
    const T *protos,
    int batch_idx,
    int mask_idx,
    int y,
    int x,
    int num_masks,
    int proto_h,
    int proto_w)
{
    int idx = batch_idx * num_masks * proto_h * proto_w +
              mask_idx * proto_h * proto_w +
              y * proto_w +
              x;
    return static_cast<float>(protos[idx]);
}

template <typename T>
__global__ void decode_filter_kernel(
    const T *input,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_masks,
    bool anchors_first,
    bool apply_sigmoid,
    float conf_thresh,
    int max_candidates,
    float *sort_keys,
    Candidate *sort_candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_anchors;
    if (idx >= total)
        return;

    int batch_idx  = idx / num_anchors;
    int anchor_idx = idx % num_anchors;
    int num_channels = num_classes + 4 + num_masks;

    // 先做类别 argmax + 置信度过滤：被阈值丢弃的 anchor（通常占绝大多数）
    // 不再读取框坐标，省掉 4 次读取。
    float max_logit = -FLT_MAX;
    int class_id    = 0;

    for (int c = 0; c < num_classes; ++c)
    {
        float logit = read_output0(
            input, batch_idx, anchor_idx, 4 + c,
            num_anchors, num_channels, anchors_first);
        if (logit > max_logit)
        {
            max_logit = logit;
            class_id  = c;
        }
    }

    float score = max_logit;
    if (apply_sigmoid)
    {
        score = 1.0f / (1.0f + expf(-score));
    }

    if (score < conf_thresh)
        return;

    int pos = atomicAdd(counts + batch_idx, 1);
    if (pos >= max_candidates)
        return;

    float cx = read_output0(input, batch_idx, anchor_idx, 0, num_anchors, num_channels, anchors_first);
    float cy = read_output0(input, batch_idx, anchor_idx, 1, num_anchors, num_channels, anchors_first);
    float w  = read_output0(input, batch_idx, anchor_idx, 2, num_anchors, num_channels, anchors_first);
    float h  = read_output0(input, batch_idx, anchor_idx, 3, num_anchors, num_channels, anchors_first);

    Candidate cand;
    cand.x1       = cx - w * 0.5f;
    cand.y1       = cy - h * 0.5f;
    cand.x2       = cx + w * 0.5f;
    cand.y2       = cy + h * 0.5f;
    cand.score    = score;
    cand.class_id = class_id;
    cand.batch_idx = batch_idx;
    cand.anchor_idx = anchor_idx;

    // 直接写入 CUB 排序输入缓冲区，省掉中间 candidates 缓冲和 prepare_sort 拷贝
    int out_idx = batch_idx * max_candidates + pos;
    sort_keys[out_idx]       = score;
    sort_candidates[out_idx] = cand;
}

struct CandidateScoreGreater
{
    __device__ __forceinline__ bool operator()(
        const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score;
    }
};

__global__ void nms_kernel(
    const Candidate *candidates,
    const int *counts,
    int total_images,
    int max_candidates,
    int max_detections,
    float iou_thresh,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes,
    int *det_to_cand_idx)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_images)
        return;

    // counts 可能因 atomicAdd 超过 max_candidates（超过部分未写入缓冲区），这里封顶
    int count = min(counts[b], max_candidates);
    if (count <= 0)
    {
        num_dets[b] = 0;
        return;
    }

    const Candidate *cand = candidates + b * max_candidates;

    float *boxes_b   = boxes + b * max_detections * 4;
    float *scores_b  = scores + b * max_detections;
    int *classes_b   = classes + b * max_detections;
    int *det_idx_b   = det_to_cand_idx + b * max_detections;

    int kept = 0;
    for (int i = 0; i < count && kept < max_detections; ++i)
    {
        const Candidate &c = cand[i];

        bool keep = true;
        for (int j = 0; j < kept; ++j)
        {
            if (classes_b[j] != c.class_id)
                continue;

            float iou_val = iou(
                c.x1, c.y1, c.x2, c.y2,
                boxes_b[j * 4 + 0],
                boxes_b[j * 4 + 1],
                boxes_b[j * 4 + 2],
                boxes_b[j * 4 + 3]);

            if (iou_val > iou_thresh)
            {
                keep = false;
                break;
            }
        }

        if (keep)
        {
            boxes_b[kept * 4 + 0] = c.x1;
            boxes_b[kept * 4 + 1] = c.y1;
            boxes_b[kept * 4 + 2] = c.x2;
            boxes_b[kept * 4 + 3] = c.y2;
            scores_b[kept]        = c.score;
            classes_b[kept]       = c.class_id;
            det_idx_b[kept]       = i;
            ++kept;
        }
    }

    num_dets[b] = kept;
}

// 由 counts 计算 CUB 分段排序的 [begin, end) 偏移：
// 每段只覆盖实际候选（段间空隙不参与排序，CUB 保证不改动空隙元素），
// 避免对 max_candidates 固定窗口做全量排序，也省去候选缓冲区的初始化。
__global__ void compute_segment_offsets_kernel(
    const int *counts,
    int total_images,
    int max_candidates,
    int *begin_offsets,
    int *end_offsets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_images)
        return;

    int begin = idx * max_candidates;
    begin_offsets[idx] = begin;
    end_offsets[idx]   = begin + min(counts[idx], max_candidates);
}

// 阶段 1：将 mask 系数从 output0 收集到连续矩阵 [max_batch*max_detections, num_masks]
// 仅填充有效检测，无效位置保持 0（不会被 GEMM 使用）
template <typename T>
__global__ void gather_coefficients_kernel(
    const T *input,
    const Candidate *candidates,
    const int *num_dets,
    const int *det_to_cand_idx,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_masks,
    int max_detections,
    int max_candidates,
    bool anchors_first,
    float *coefficients)
{
    int b = blockIdx.y;
    int det = blockIdx.x;

    if (b >= total_images || det >= num_dets[b])
        return;

    int cand_idx = det_to_cand_idx[b * max_detections + det];
    int anchor_idx = candidates[b * max_candidates + cand_idx].anchor_idx;
    int num_channels = num_classes + 4 + num_masks;

    float *dst = coefficients + (b * max_detections + det) * num_masks;

    for (int i = threadIdx.x; i < num_masks; i += blockDim.x)
    {
        dst[i] = read_output0(
            input, b, anchor_idx, 4 + num_classes + i,
            num_anchors, num_channels, anchors_first);
    }
}

// 阶段 1 辅助：将 FP16 prototype 转换为 FP32，供 cuBLAS 使用
__global__ void convert_fp16_to_fp32_kernel(
    const __half *src,
    float *dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        dst[idx] = __half2float(src[idx]);
    }
}

// 阶段 2：从 raw_masks 裁剪并 resize 到 160x160，同时设置 mask 元数据
// 每个 block 处理一个 (batch, det)，每个线程处理一个输出像素
__global__ void crop_resize_masks_kernel(
    const float *raw_masks,
    const float *boxes,
    const int *num_dets,
    int total_images,
    int proto_h,
    int proto_w,
    int input_width,
    int input_height,
    int max_detections,
    float *detection_masks,
    int *mask_offsets,
    int *mask_shapes)
{
    constexpr int MASK_SIZE = kMaskOutputSize;
    int b = blockIdx.y;
    int det = blockIdx.x;

    if (b >= total_images || det >= max_detections)
        return;

    int nd = num_dets[b];
    int out_idx = b * max_detections * MASK_SIZE * MASK_SIZE + det * MASK_SIZE * MASK_SIZE;
    mask_offsets[b * max_detections + det] = out_idx;
    mask_shapes[(b * max_detections + det) * 2 + 0] = MASK_SIZE;
    mask_shapes[(b * max_detections + det) * 2 + 1] = MASK_SIZE;

    if (det >= nd)
    {
        // init_output_kernel 已将整个 mask 缓冲区清零
        return;
    }

    const float *box = boxes + (b * max_detections + det) * 4;

    float scale_x = static_cast<float>(proto_w) / static_cast<float>(input_width);
    float scale_y = static_cast<float>(proto_h) / static_cast<float>(input_height);

    float x1p = box[0] * scale_x;
    float y1p = box[1] * scale_y;
    float x2p = box[2] * scale_x;
    float y2p = box[3] * scale_y;

    int x1 = max(0, min(proto_w - 1, (int)floorf(x1p)));
    int y1 = max(0, min(proto_h - 1, (int)floorf(y1p)));
    int x2 = max(0, min(proto_w, (int)ceilf(x2p)));
    int y2 = max(0, min(proto_h, (int)ceilf(y2p)));

    int crop_w = x2 - x1;
    int crop_h = y2 - y1;

    float *dst = detection_masks + out_idx;

    if (crop_w <= 0 || crop_h <= 0)
    {
        for (int tid = threadIdx.x; tid < MASK_SIZE * MASK_SIZE; tid += blockDim.x)
        {
            dst[tid] = 0.0f;
        }
        return;
    }

    const float *src = raw_masks + (b * max_detections + det) * proto_h * proto_w;

    for (int tid = threadIdx.x; tid < MASK_SIZE * MASK_SIZE; tid += blockDim.x)
    {
        int oy = tid / MASK_SIZE;
        int ox = tid % MASK_SIZE;

        float px = x1 + (ox + 0.5f) * static_cast<float>(crop_w) / MASK_SIZE;
        float py = y1 + (oy + 0.5f) * static_cast<float>(crop_h) / MASK_SIZE;

        int sx = min(proto_w - 1, max(0, (int)floorf(px)));
        int sy = min(proto_h - 1, max(0, (int)floorf(py)));

        float val = src[sy * proto_w + sx];
        dst[tid] = 1.0f / (1.0f + expf(-val));
    }
}

void yolo11_seg_compute_masks_gpu(
    const void *input,
    const void *mask_protos,
    bool input_is_half,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_masks,
    int proto_h,
    int proto_w,
    int input_width,
    int input_height,
    bool anchors_first,
    const int *d_num_dets,
    const int *d_det_to_cand_idx,
    const Candidate *d_candidates,
    int max_detections,
    int max_candidates,
    const float *d_boxes,
    float *d_detection_masks,
    int *d_mask_offsets,
    int *d_mask_shapes,
    cublasHandle_t cublas_handle,
    float *d_coefficients,
    float *d_raw_masks,
    float *d_proto_fp32,
    int *h_num_dets,
    cudaStream_t stream)
{
    if (total_images <= 0)
        return;

    const int proto_pixels = proto_h * proto_w;

    // 1. 读取 num_dets 到 CPU，用于按 batch 调用 GEMM
    checkRuntime(cudaMemcpyAsync(h_num_dets, d_num_dets, total_images * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // 2. 收集系数到连续矩阵 [max_batch*max_detections, num_masks]
    dim3 grid_coeffs(max_detections, total_images);
    if (input_is_half)
    {
        gather_coefficients_kernel<<<grid_coeffs, 256, 0, stream>>>(
            static_cast<const __half *>(input),
            d_candidates, d_num_dets, d_det_to_cand_idx,
            total_images, num_anchors, num_classes, num_masks,
            max_detections, max_candidates, anchors_first,
            d_coefficients);
    }
    else
    {
        gather_coefficients_kernel<<<grid_coeffs, 256, 0, stream>>>(
            static_cast<const float *>(input),
            d_candidates, d_num_dets, d_det_to_cand_idx,
            total_images, num_anchors, num_classes, num_masks,
            max_detections, max_candidates, anchors_first,
            d_coefficients);
    }
    checkRuntime(cudaPeekAtLastError());

    // 3. 准备 FP32 prototype
    const int proto_total = total_images * num_masks * proto_pixels;
    const float *d_proto_fp32_ptr = nullptr;
    if (input_is_half)
    {
        convert_fp16_to_fp32_kernel<<<(proto_total + 255) / 256, 256, 0, stream>>>(
            static_cast<const __half *>(mask_protos),
            d_proto_fp32,
            proto_total);
        checkRuntime(cudaPeekAtLastError());
        d_proto_fp32_ptr = d_proto_fp32;
    }
    else
    {
        d_proto_fp32_ptr = static_cast<const float *>(mask_protos);
    }

    // 4. 按 batch 调用 cuBLAS GEMM：
    //    raw_masks_b = coefficients_b [N_b, num_masks] × proto_b [num_masks, proto_pixels]
    // 使用 CUBLAS_OP_N, N 配合 row-major 数据的 reinterpretation，得到 row-major 输出。
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int b = 0; b < total_images; ++b)
    {
        int n_b = h_num_dets[b];
        if (n_b <= 0)
            continue;

        const float *coeffs_b = d_coefficients + b * max_detections * num_masks;
        const float *proto_b = d_proto_fp32_ptr + b * num_masks * proto_pixels;
        float *raw_masks_b = d_raw_masks + b * max_detections * proto_pixels;

        cublasStatus_t status = cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            proto_pixels, n_b, num_masks,
            &alpha,
            proto_b, proto_pixels,
            coeffs_b, num_masks,
            &beta,
            raw_masks_b, proto_pixels);

        if (status != CUBLAS_STATUS_SUCCESS)
        {
            fprintf(stderr, "[yolo11_seg_compute_masks_gpu] cublasSgemm failed for batch %d: %d\n", b, status);
        }
    }

    // 5. 裁剪并 resize 到 160x160
    dim3 grid_crop(max_detections, total_images);
    crop_resize_masks_kernel<<<grid_crop, 256, 0, stream>>>(
        d_raw_masks,
        d_boxes,
        d_num_dets,
        total_images,
        proto_h,
        proto_w,
        input_width,
        input_height,
        max_detections,
        d_detection_masks,
        d_mask_offsets,
        d_mask_shapes);
    checkRuntime(cudaPeekAtLastError());
}

__global__ void init_output_kernel(
    int total_images,
    int max_detections,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes,
    float *detection_masks,
    int *mask_offsets,
    int *mask_shapes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * max_detections;
    if (idx >= total)
        return;

    boxes[idx * 4 + 0] = 0.0f;
    boxes[idx * 4 + 1] = 0.0f;
    boxes[idx * 4 + 2] = 0.0f;
    boxes[idx * 4 + 3] = 0.0f;
    scores[idx]        = 0.0f;
    classes[idx]       = -1;
    mask_offsets[idx]  = -1;
    mask_shapes[idx * 2 + 0] = 0;
    mask_shapes[idx * 2 + 1] = 0;

    constexpr int MASK_SIZE = kMaskOutputSize;
    int mask_total = total_images * max_detections * MASK_SIZE * MASK_SIZE;
    for (int i = idx; i < mask_total; i += total)
    {
        detection_masks[i] = 0.0f;
    }

    if (idx < total_images)
        num_dets[idx] = 0;
}

size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments)
{
    size_t bytes = 0;
    float *d_keys_in = nullptr;
    float *d_keys_out = nullptr;
    Candidate *d_candidates_in = nullptr;
    Candidate *d_candidates_out = nullptr;
    int *d_begin_offsets = nullptr;
    int *d_end_offsets = nullptr;

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, bytes,
        d_keys_in, d_keys_out,
        d_candidates_in, d_candidates_out,
        total_candidates, num_segments,
        d_begin_offsets, d_end_offsets,
        0, sizeof(float) * 8);

    return bytes;
}

void yolo11_seg_postprocess_gpu(
    const void *input,
    const void *mask_protos,
    bool input_is_half,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_masks,
    int proto_h,
    int proto_w,
    int input_width,
    int input_height,
    bool anchors_first,
    bool apply_sigmoid,
    float conf_thresh,
    float iou_thresh,
    int max_detections,
    int max_candidates,
    int *d_counts,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_detection_masks,
    int *d_mask_offsets,
    int *d_mask_shapes,
    int *d_det_to_cand_idx,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_begin_offsets,
    int *d_sort_end_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_anchors <= 0)
        return;

    const int block = 256;

    // 初始化候选计数和输出缓冲区
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));

    int total_out = total_images * max_detections;
    int grid_out  = (total_out + block - 1) / block;
    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes,
        d_detection_masks, d_mask_offsets, d_mask_shapes);
    checkRuntime(cudaPeekAtLastError());

    // decode + 置信度过滤，候选直接写入 CUB 排序输入缓冲区
    int total_threads = total_images * num_anchors;
    int grid  = (total_threads + block - 1) / block;

    if (input_is_half)
    {
        decode_filter_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half *>(input),
            total_images, num_anchors, num_classes, num_masks,
            anchors_first, apply_sigmoid, conf_thresh, max_candidates,
            d_sort_keys_in, d_sort_candidates_in, d_counts);
    }
    else
    {
        decode_filter_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float *>(input),
            total_images, num_anchors, num_classes, num_masks,
            anchors_first, apply_sigmoid, conf_thresh, max_candidates,
            d_sort_keys_in, d_sort_candidates_in, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());

    // 按实际候选数计算分段偏移，CUB 只排有效候选（段间空隙不参与排序）
    compute_segment_offsets_kernel<<<(total_images + block - 1) / block, block, 0, stream>>>(
        d_counts, total_images, max_candidates,
        d_sort_begin_offsets, d_sort_end_offsets);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_temp, cub_temp_storage_bytes,
        d_sort_keys_in, d_sort_keys_out,
        d_sort_candidates_in, d_sort_candidates_out,
        total_images * max_candidates,
        total_images,
        d_sort_begin_offsets, d_sort_end_offsets,
        0, sizeof(float) * 8,
        stream));

    // NMS 直接读取排序输出
    nms_kernel<<<(total_images + block - 1) / block, block, 0, stream>>>(
        d_sort_candidates_out, d_counts, total_images, max_candidates, max_detections,
        iou_thresh, d_num_dets, d_boxes, d_scores, d_classes, d_det_to_cand_idx);
    checkRuntime(cudaPeekAtLastError());

    // mask 计算已拆分到 yolo11_seg_compute_masks_gpu 中，使用 cuBLAS GEMM + 轻量裁剪核。
}
}

} // namespace yolo11_seg_postprocess
