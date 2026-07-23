/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_seg_postprocess/yolo26_seg_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <cub/cub.cuh>

namespace yolo26_seg_postprocess
{

template <typename T>
static __device__ __forceinline__ float read_input(
    const T *input, int idx)
{
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

__global__ void init_candidates_kernel(
    int total_images,
    int num_predictions,
    Candidate *candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_predictions;
    if (idx >= total)
        return;

    candidates[idx].score    = -FLT_MAX;
    candidates[idx].class_id = -1;
    candidates[idx].batch_idx = idx / num_predictions;
    candidates[idx].pred_idx  = idx % num_predictions;
}

template <typename T>
__global__ void filter_kernel(
    const T *input,
    int total_images,
    int num_predictions,
    int num_masks,
    float conf_thresh,
    Candidate *candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_predictions;
    if (idx >= total)
        return;

    int batch_idx = idx / num_predictions;
    int pred_idx  = idx % num_predictions;

    int num_fields = 6 + num_masks;
    int offset = (batch_idx * num_predictions + pred_idx) * num_fields;

    float x1    = read_input(input, offset + 0);
    float y1    = read_input(input, offset + 1);
    float x2    = read_input(input, offset + 2);
    float y2    = read_input(input, offset + 3);
    float score = read_input(input, offset + 4);
    int class_id = __float2int_rn(read_input(input, offset + 5));

    if (score < conf_thresh)
        return;

    int pos = atomicAdd(counts + batch_idx, 1);
    if (pos >= num_predictions)
        return;

    Candidate cand;
    cand.x1       = x1;
    cand.y1       = y1;
    cand.x2       = x2;
    cand.y2       = y2;
    cand.score    = score;
    cand.class_id = class_id;
    cand.batch_idx = batch_idx;
    cand.pred_idx  = pred_idx;

    candidates[batch_idx * num_predictions + pos] = cand;
}

struct CandidateScoreGreater
{
    __device__ __forceinline__ bool operator()(const Candidate &a, const Candidate &b) const
    {
        return a.score > b.score;
    }
};

__global__ void prepare_sort_kernel(
    const Candidate *candidates,
    int total,
    float *keys_out,
    Candidate *values_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;

    keys_out[idx]   = candidates[idx].score;
    values_out[idx] = candidates[idx];
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

__global__ void write_topk_kernel(
    const Candidate *candidates,
    const int *counts,
    int total_images,
    int num_predictions,
    int max_detections,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_images)
        return;

    int count = counts[b];
    if (count > num_predictions)
        count = num_predictions;

    int keep = min(count, max_detections);
    const Candidate *cand = candidates + b * num_predictions;

    for (int i = 0; i < keep; ++i)
    {
        const Candidate &c = cand[i];
        int out_idx = b * max_detections + i;
        boxes[out_idx * 4 + 0] = c.x1;
        boxes[out_idx * 4 + 1] = c.y1;
        boxes[out_idx * 4 + 2] = c.x2;
        boxes[out_idx * 4 + 3] = c.y2;
        scores[out_idx]        = c.score;
        classes[out_idx]       = c.class_id;
    }

    num_dets[b] = keep;
}

// ============================================================
// 融合 kernel：系数读取 + matmul + crop + sigmoid 一步完成
// 仅在 crop 区域内逐像素计算 dot(mask_weights, proto[:,y,x])，
// 消除中间 raw_masks buffer 和 cuBLAS 对大框全空间的无效计算。
// ============================================================
template <typename T_INPUT, typename T_PROTO>
__global__ void compute_and_crop_masks_kernel(
    const T_INPUT *input,
    const T_PROTO *protos,
    const Candidate *candidates,
    const int *num_dets,
    const float *boxes,
    int total_images,
    int num_predictions,
    int num_masks,
    int max_detections,
    int proto_h,
    int proto_w,
    int mask_output_resolution,
    int input_width,
    int input_height,
    float *detection_masks,
    int *mask_offsets,
    int *mask_shapes)
{
    int b = blockIdx.y;
    int det = blockIdx.x;

    if (b >= total_images || det >= max_detections)
        return;

    int nd = num_dets[b];
    int out_idx = b * max_detections * mask_output_resolution * mask_output_resolution + det * mask_output_resolution * mask_output_resolution;
    mask_offsets[b * max_detections + det] = out_idx;
    mask_shapes[(b * max_detections + det) * 2 + 0] = mask_output_resolution;
    mask_shapes[(b * max_detections + det) * 2 + 1] = mask_output_resolution;

    if (det >= nd)
        return;  // mask buffer 已由 cudaMemsetAsync 清零

    // —— 读取 mask 系数（yolo26: 字段 6..5+num_masks） ——
    int pred_idx = candidates[b * num_predictions + det].pred_idx;
    int num_fields = 6 + num_masks;

    float mask_weights[32];
    for (int i = 0; i < num_masks; ++i)
    {
        mask_weights[i] = static_cast<float>(
            input[(b * num_predictions + pred_idx) * num_fields + 6 + i]);
    }

    // —— 计算 crop 区域 ——
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
        for (int tid = threadIdx.x; tid < mask_output_resolution * mask_output_resolution; tid += blockDim.x)
            dst[tid] = 0.0f;
        return;
    }

    // —— 逐像素 matmul + sigmoid ——
    const T_PROTO *proto_b = protos + static_cast<size_t>(b) * num_masks * proto_h * proto_w;
    const int proto_stride = proto_h * proto_w;

    for (int tid = threadIdx.x; tid < mask_output_resolution * mask_output_resolution; tid += blockDim.x)
    {
        int oy = tid / mask_output_resolution;
        int ox = tid % mask_output_resolution;

        float px = x1p + static_cast<float>(ox) * (x2p - x1p) / static_cast<float>(mask_output_resolution);
        float py = y1p + static_cast<float>(oy) * (y2p - y1p) / static_cast<float>(mask_output_resolution);

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

void yolo26_seg_compute_masks_gpu(
    const void *input,
    const void *mask_protos,
    bool input_is_half,
    int total_images,
    int num_predictions,
    int num_masks,
    int proto_h,
    int proto_w,
    int mask_output_resolution,
    int input_width,
    int input_height,
    const int *d_num_dets,
    const float *d_boxes,
    const Candidate *d_candidates,
    int max_detections,
    float *d_detection_masks,
    int *d_mask_offsets,
    int *d_mask_shapes,
    cudaStream_t stream)
{
    if (total_images <= 0)
        return;
    dim3 grid(max_detections, total_images);
    if (input_is_half)
    {
        compute_and_crop_masks_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const __half *>(input),
            static_cast<const __half *>(mask_protos),
            d_candidates, d_num_dets,
            d_boxes,
            total_images, num_predictions, num_masks,
            max_detections,
            proto_h, proto_w, mask_output_resolution, input_width, input_height,
            d_detection_masks, d_mask_offsets, d_mask_shapes);
    }
    else
    {
        compute_and_crop_masks_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const float *>(input),
            static_cast<const float *>(mask_protos),
            d_candidates, d_num_dets,
            d_boxes,
            total_images, num_predictions, num_masks,
            max_detections,
            proto_h, proto_w, mask_output_resolution, input_width, input_height,
            d_detection_masks, d_mask_offsets, d_mask_shapes);
    }
    checkRuntime(cudaPeekAtLastError());
}

size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments)
{
    size_t bytes = 0;
    float *d_keys_in = nullptr;
    float *d_keys_out = nullptr;
    Candidate *d_candidates_in = nullptr;
    Candidate *d_candidates_out = nullptr;
    int *d_offsets = nullptr;

    cub::DeviceSegmentedRadixSort::SortPairsDescending(
        nullptr, bytes,
        d_keys_in, d_keys_out,
        d_candidates_in, d_candidates_out,
        total_candidates, num_segments,
        d_offsets, d_offsets + 1,
        0, sizeof(float) * 8);

    return bytes;
}

void yolo26_seg_postprocess_gpu(
    const void *input,
    const void *mask_protos,
    bool input_is_half,
    int total_images,
    int num_predictions,
    int num_masks,
    int proto_h,
    int proto_w,
    int mask_output_resolution,
    int input_width,
    int input_height,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_detection_masks,
    int *d_mask_offsets,
    int *d_mask_shapes,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_predictions <= 0)
        return;

    const int block = 256;
    const int total_cand = total_images * num_predictions;
    const int grid_cand = (total_cand + block - 1) / block;
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));
    init_candidates_kernel<<<grid_cand, block, 0, stream>>>(
        total_images, num_predictions, d_candidates);
    checkRuntime(cudaPeekAtLastError());
    const int total_work = total_images * num_predictions;
    const int grid_filter = (total_work + block - 1) / block;
    if (input_is_half)
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const half *>(input),
            total_images, num_predictions, num_masks,
            conf_thresh, d_candidates, d_counts);
    }
    else
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const float *>(input),
            total_images, num_predictions, num_masks,
            conf_thresh, d_candidates, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());
    int sort_total = total_images * num_predictions;
    int grid_sort = (sort_total + block - 1) / block;
    prepare_sort_kernel<<<grid_sort, block, 0, stream>>>(
        d_candidates, sort_total,
        d_sort_keys_in, d_sort_candidates_in);
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cub::DeviceSegmentedRadixSort::SortPairsDescending(
        d_cub_temp, cub_temp_storage_bytes,
        d_sort_keys_in, d_sort_keys_out,
        d_sort_candidates_in, d_sort_candidates_out,
        sort_total,
        total_images,
        d_sort_offsets, d_sort_offsets + 1,
        0, sizeof(float) * 8,
        stream));
    checkRuntime(cudaMemcpyAsync(
        d_candidates, d_sort_candidates_out,
        sort_total * sizeof(Candidate),
        cudaMemcpyDeviceToDevice, stream));
    const int total_out = total_images * max_detections;
    const int grid_out = (total_out + block - 1) / block;

    // mask buffer 用硬件加速清零
    size_t mask_total = static_cast<size_t>(total_images) * max_detections * mask_output_resolution * mask_output_resolution;
    checkRuntime(cudaMemsetAsync(d_detection_masks, 0, mask_total * sizeof(float), stream));

    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes,
        d_mask_offsets, d_mask_shapes);
    checkRuntime(cudaPeekAtLastError());
    const int grid_batch = (total_images + block - 1) / block;
    write_topk_kernel<<<grid_batch, block, 0, stream>>>(
        d_candidates, d_counts,
        total_images, num_predictions, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());

    // mask 计算已拆分到 yolo26_seg_compute_masks_gpu 中，使用融合 kernel 一步完成。
}

} // namespace yolo26_seg_postprocess
