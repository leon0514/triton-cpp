/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_seg_postprocess/yolo11_seg_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// ============================================================
// 融合 kernel：gather 系数 + matmul + crop + sigmoid 一步完成
// 仿参考实现 decode_single_mask，但以单一 grid 启动，避免
// per-detection launch overhead 和中间 raw_masks buffer。
// 仅在 crop 区域内逐像素计算 dot(mask_weights, proto[:,y,x])，
// 大幅减少无效计算（小框只算 ~900 像素 vs cuBLAS 的 25600）。
// ============================================================
template <typename T_INPUT, typename T_PROTO>
__global__ void compute_and_crop_masks_kernel(
    const T_INPUT *input,
    const T_PROTO *protos,
    const Candidate *candidates,
    const int *num_dets,
    const int *det_to_cand_idx,
    const float *boxes,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_masks,
    int max_detections,
    int max_candidates,
    int proto_h,
    int proto_w,
    int input_width,
    int input_height,
    bool anchors_first,
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
        return;  // mask buffer 已由 cudaMemsetAsync 清零

    // —— 读取 mask 系数（每个线程独立读取全部 32 个，warp 内 L1 共享） ——
    int cand_idx   = det_to_cand_idx[b * max_detections + det];
    int anchor_idx = candidates[b * max_candidates + cand_idx].anchor_idx;
    int num_channels = num_classes + 4 + num_masks;

    float mask_weights[32];  // num_masks ≤ 32
    for (int i = 0; i < num_masks; ++i)
    {
        mask_weights[i] = read_output0(
            input, b, anchor_idx, 4 + num_classes + i,
            num_anchors, num_channels, anchors_first);
    }

    // —— 计算 crop 区域在 proto 空间中的坐标 ——
    const float *box = boxes + (b * max_detections + det) * 4;
    float scale_x = static_cast<float>(proto_w) / static_cast<float>(input_width);
    float scale_y = static_cast<float>(proto_h) / static_cast<float>(input_height);

    float x1p = box[0] * scale_x;
    float y1p = box[1] * scale_y;
    float x2p = box[2] * scale_x;
    float y2p = box[3] * scale_y;

    // 钳位到 proto 有效范围，保留浮点精度，采样时不再 floor/ceil 取整
    x1p = fmaxf(0.0f, fminf(static_cast<float>(proto_w), x1p));
    y1p = fmaxf(0.0f, fminf(static_cast<float>(proto_h), y1p));
    x2p = fmaxf(0.0f, fminf(static_cast<float>(proto_w), x2p));
    y2p = fmaxf(0.0f, fminf(static_cast<float>(proto_h), y2p));

    float *dst = detection_masks + out_idx;

    if (x2p - x1p <= 0.0f || y2p - y1p <= 0.0f)
    {
        for (int tid = threadIdx.x; tid < MASK_SIZE * MASK_SIZE; tid += blockDim.x)
            dst[tid] = 0.0f;
        return;
    }

    // —— 逐输出像素：采样 proto → dot(mask_weights, proto[:,sy,sx]) → sigmoid ——
    // proto 排布: [batch, mask, H, W]，同 mask 通道内连续，跨通道 stride=H*W
    const T_PROTO *proto_b = protos + static_cast<size_t>(b) * num_masks * proto_h * proto_w;
    const int proto_stride = proto_h * proto_w;

    for (int tid = threadIdx.x; tid < MASK_SIZE * MASK_SIZE; tid += blockDim.x)
    {
        int oy = tid / MASK_SIZE;
        int ox = tid % MASK_SIZE;

        float px = x1p + static_cast<float>(ox) * (x2p - x1p) / static_cast<float>(MASK_SIZE);
        float py = y1p + static_cast<float>(oy) * (y2p - y1p) / static_cast<float>(MASK_SIZE);

        // —— 双线性插值：计算 dot(w, proto) 在 4 个角点，再插值 ——
        int sx0 = static_cast<int>(floorf(px));
        int sy0 = static_cast<int>(floorf(py));
        int sx1 = min(sx0 + 1, proto_w - 1);
        int sy1 = min(sy0 + 1, proto_h - 1);
        sx0 = max(sx0, 0);
        sy0 = max(sy0, 0);

        float fx = px - static_cast<float>(sx0);
        float fy = py - static_cast<float>(sy0);

        float c00 = 0.0f, c10 = 0.0f, c01 = 0.0f, c11 = 0.0f;
        for (int ic = 0; ic < num_masks; ++ic)
        {
            float p00 = static_cast<float>(proto_b[ic * proto_stride + sy0 * proto_w + sx0]);
            float p10 = static_cast<float>(proto_b[ic * proto_stride + sy0 * proto_w + sx1]);
            float p01 = static_cast<float>(proto_b[ic * proto_stride + sy1 * proto_w + sx0]);
            float p11 = static_cast<float>(proto_b[ic * proto_stride + sy1 * proto_w + sx1]);

            float w = mask_weights[ic];
            c00 += p00 * w;
            c10 += p10 * w;
            c01 += p01 * w;
            c11 += p11 * w;
        }

        float cumprod = (c00 * (1.0f - fx) + c10 * fx) * (1.0f - fy) +
                        (c01 * (1.0f - fx) + c11 * fx) * fy;

        dst[tid] = 1.0f / (1.0f + expf(-cumprod));
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
    cudaStream_t stream)
{
    if (total_images <= 0)
        return;
    // 单次 kernel launch：系数读取 + matmul + crop + sigmoid 全部融合
    dim3 grid(max_detections, total_images);
    if (input_is_half)
    {
        compute_and_crop_masks_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const __half *>(input),
            static_cast<const __half *>(mask_protos),
            d_candidates, d_num_dets, d_det_to_cand_idx,
            d_boxes,
            total_images, num_anchors, num_classes, num_masks,
            max_detections, max_candidates,
            proto_h, proto_w, input_width, input_height,
            anchors_first,
            d_detection_masks, d_mask_offsets, d_mask_shapes);
    }
    else
    {
        compute_and_crop_masks_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const float *>(input),
            static_cast<const float *>(mask_protos),
            d_candidates, d_num_dets, d_det_to_cand_idx,
            d_boxes,
            total_images, num_anchors, num_classes, num_masks,
            max_detections, max_candidates,
            proto_h, proto_w, input_width, input_height,
            anchors_first,
            d_detection_masks, d_mask_offsets, d_mask_shapes);
    }
    checkRuntime(cudaPeekAtLastError());
}

__global__ void init_output_kernel(
    int total_images,
    int max_detections,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes,
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

    // mask buffer 用硬件加速清零，远快于 kernel 逐元素写
    constexpr int MASK_SIZE = kMaskOutputSize;
    size_t mask_total = static_cast<size_t>(total_images) * max_detections * MASK_SIZE * MASK_SIZE;
    checkRuntime(cudaMemsetAsync(d_detection_masks, 0, mask_total * sizeof(float), stream));

    int total_out = total_images * max_detections;
    int grid_out  = (total_out + block - 1) / block;
    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes,
        d_mask_offsets, d_mask_shapes);
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

    // mask 计算已拆分到 yolo11_seg_compute_masks_gpu 中，使用融合 kernel 一步完成。
}

} // namespace yolo11_seg_postprocess
