/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "rfdetr_seg_postprocess/rfdetr_seg_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <cub/cub.cuh>

namespace rfdetr_seg_postprocess
{

// RF-DETR ONNX 输出 labels 维度为 91：
//   - dim 0 为背景/无目标槽位
//   - dim 1..90 对应官方 COCO ID 1..90
// COCO 90 个 ID 中有 10 个空 ID，由 config.pbtxt 中的 skip_coco_ids 指定。
// 后处理跳过空类别，输出 class_id 为 names 文件中的 0-based 索引。

__device__ __forceinline__ float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

template <typename T>
static __device__ __forceinline__ float read_input(const void *input, int idx)
{
    return static_cast<float>(reinterpret_cast<const T *>(input)[idx]);
}

__global__ void init_candidates_kernel(
    int total_images,
    int num_queries,
    Candidate *candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_queries;
    if (idx >= total)
        return;

    candidates[idx].score     = -FLT_MAX;
    candidates[idx].class_id  = -1;
    candidates[idx].batch_idx = idx / num_queries;
    candidates[idx].query_idx = idx % num_queries;
}

template <typename T>
__global__ void filter_kernel(
    const T *dets,
    const T *labels,
    int total_images,
    int num_queries,
    float input_width,
    float input_height,
    float conf_thresh,
    const int *coco_id_to_index,
    Candidate *candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_queries;
    if (idx >= total)
        return;

    int batch_idx = idx / num_queries;
    int query_idx = idx % num_queries;

    // dets 读取: [batch, num_queries, 4] 归一化 cxcywh
    int det_offset = (batch_idx * num_queries + query_idx) * 4;
    float cx = static_cast<float>(dets[det_offset + 0]);
    float cy = static_cast<float>(dets[det_offset + 1]);
    float w  = static_cast<float>(dets[det_offset + 2]);
    float h  = static_cast<float>(dets[det_offset + 3]);

    // labels 读取: [batch, num_queries, 91]，dim 0 为背景，dim 1..90 为 COCO ID 1~90
    int label_offset = (batch_idx * num_queries + query_idx) * 91;

    // 对有效 COCO ID 做 sigmoid，取最大值与索引
    float best_score  = 0.0f;
    int best_class_id = -1;  // names 文件中的 0-based 索引
    #pragma unroll 4
    for (int coco_id = 1; coco_id <= 90; ++coco_id)
    {
        int class_idx = coco_id_to_index[coco_id];
        if (class_idx < 0)
            continue;

        float logit = static_cast<float>(labels[label_offset + coco_id]);
        float score = sigmoid(logit);
        if (score > best_score)
        {
            best_score  = score;
            best_class_id = class_idx;
        }
    }

    if (best_class_id < 0)
        return;

    if (best_score < conf_thresh)
        return;

    // 反归一化到模型输入坐标系，并转换为 xyxy
    float x1 = (cx - 0.5f * w) * input_width;
    float y1 = (cy - 0.5f * h) * input_height;
    float x2 = (cx + 0.5f * w) * input_width;
    float y2 = (cy + 0.5f * h) * input_height;

    int pos = atomicAdd(counts + batch_idx, 1);
    if (pos >= num_queries)
        return;

    Candidate cand;
    cand.x1       = x1;
    cand.y1       = y1;
    cand.x2       = x2;
    cand.y2       = y2;
    cand.score    = best_score;
    cand.class_id = best_class_id;
    cand.batch_idx = batch_idx;
    cand.query_idx = query_idx;

    candidates[batch_idx * num_queries + pos] = cand;
}

// 从候选框中提取 score/key 并复制 value，供 CUB 分段排序使用
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
    int *det_to_query_idx,
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
    det_to_query_idx[idx] = -1;

    if (mask_offsets != nullptr)
        mask_offsets[idx] = -1;
    if (mask_shapes != nullptr)
    {
        mask_shapes[idx * 2 + 0] = 0;
        mask_shapes[idx * 2 + 1] = 0;
    }

    if (idx < total_images)
        num_dets[idx] = 0;
}

__global__ void write_topk_kernel(
    const Candidate *candidates,
    const int *counts,
    int total_images,
    int num_queries,
    int max_detections,
    int *num_dets,
    float *boxes,
    float *scores,
    int *classes,
    int *det_to_query_idx)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= total_images)
        return;

    int count = counts[b];
    if (count > num_queries)
        count = num_queries;

    int keep = min(count, max_detections);
    const Candidate *cand = candidates + b * num_queries;

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
        det_to_query_idx[out_idx] = c.query_idx;
    }

    num_dets[b] = keep;
}

template <typename T>
__global__ void compute_masks_kernel(
    const T *masks,  // [batch, num_queries, mask_h, mask_w]
    const float *boxes,
    const int *num_dets,
    const int *det_to_query_idx,
    int total_images,
    int num_queries,
    int max_detections,
    float input_width,
    float input_height,
    int mask_height,
    int mask_width,
    int mask_output_resolution,
    float *detection_masks,
    int *mask_offsets,
    int *mask_shapes)
{
    int b = blockIdx.y;
    int k = blockIdx.x;

    if (b >= total_images || k >= max_detections)
        return;

    int idx = b * max_detections + k;

    int nd = num_dets[b];

    int out_idx = idx * mask_output_resolution * mask_output_resolution;
    mask_offsets[idx] = out_idx;
    mask_shapes[idx * 2 + 0] = mask_output_resolution;
    mask_shapes[idx * 2 + 1] = mask_output_resolution;

    if (k >= nd)
        return;

    int query_idx = det_to_query_idx[idx];
    if (query_idx < 0)
        return;

    const float *box = boxes + idx * 4;
    float x1 = box[0];
    float y1 = box[1];
    float x2 = box[2];
    float y2 = box[3];

    float sx = static_cast<float>(mask_width) / input_width;
    float sy = static_cast<float>(mask_height) / input_height;

    float m_x1 = x1 * sx;
    float m_y1 = y1 * sy;
    float m_x2 = x2 * sx;
    float m_y2 = y2 * sy;

    // 钳位到 mask 有效范围，保留浮点精度，采样时不再 floor/ceil 取整
    m_x1 = fmaxf(0.0f, fminf(static_cast<float>(mask_width), m_x1));
    m_y1 = fmaxf(0.0f, fminf(static_cast<float>(mask_height), m_y1));
    m_x2 = fmaxf(0.0f, fminf(static_cast<float>(mask_width), m_x2));
    m_y2 = fmaxf(0.0f, fminf(static_cast<float>(mask_height), m_y2));

    float *out = detection_masks + out_idx;

    if (m_x2 - m_x1 <= 0.0f || m_y2 - m_y1 <= 0.0f)
    {
        for (int tid = threadIdx.x; tid < mask_output_resolution * mask_output_resolution; tid += blockDim.x)
        {
            out[tid] = 0.0f;
        }
        return;
    }

    const T *src = masks + ((b * num_queries + query_idx) * mask_height * mask_width);

    for (int tid = threadIdx.x; tid < mask_output_resolution * mask_output_resolution; tid += blockDim.x)
    {
        int oy = tid / mask_output_resolution;
        int ox = tid % mask_output_resolution;

        float px = m_x1 + static_cast<float>(ox) * (m_x2 - m_x1) / static_cast<float>(mask_output_resolution);
        float py = m_y1 + static_cast<float>(oy) * (m_y2 - m_y1) / static_cast<float>(mask_output_resolution);

        // —— 双线性插值 ——
        int mx0 = static_cast<int>(floorf(px));
        int my0 = static_cast<int>(floorf(py));
        int mx1 = min(mx0 + 1, mask_width - 1);
        int my1 = min(my0 + 1, mask_height - 1);
        mx0 = max(mx0, 0);
        my0 = max(my0, 0);

        float fx = px - static_cast<float>(mx0);
        float fy = py - static_cast<float>(my0);

        float v00 = static_cast<float>(src[my0 * mask_width + mx0]);
        float v10 = static_cast<float>(src[my0 * mask_width + mx1]);
        float v01 = static_cast<float>(src[my1 * mask_width + mx0]);
        float v11 = static_cast<float>(src[my1 * mask_width + mx1]);

        float logit = (v00 * (1.0f - fx) + v10 * fx) * (1.0f - fy) +
                      (v01 * (1.0f - fx) + v11 * fx) * fy;

        out[tid] = sigmoid(logit);
    }
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

void rfdetr_seg_postprocess_gpu(
    const void *dets,
    const void *labels,
    const void *masks,
    bool input_is_half,
    int total_images,
    int num_queries,
    float input_width,
    float input_height,
    int mask_height,
    int mask_width,
    int mask_output_resolution,
    bool parse_masks,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    int *d_det_to_query_idx,
    float *d_detection_masks,
    int *d_mask_offsets,
    int *d_mask_shapes,
    const int *d_coco_id_to_index,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_queries <= 0)
        return;

    const int block = 256;
    const int total_cand = total_images * num_queries;
    const int grid_cand = (total_cand + block - 1) / block;
    // 1. 清零计数并初始化候选缓冲区
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));
    init_candidates_kernel<<<grid_cand, block, 0, stream>>>(
        total_images, num_queries, d_candidates);
    checkRuntime(cudaPeekAtLastError());
    // 2. 置信度过滤、类别选择、坐标转换
    const int total_work = total_images * num_queries;
    const int grid_filter = (total_work + block - 1) / block;
    if (input_is_half)
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const half *>(dets),
            reinterpret_cast<const half *>(labels),
            total_images, num_queries,
            input_width, input_height,
            conf_thresh,
            d_coco_id_to_index,
            d_candidates, d_counts);
    }
    else
    {
        filter_kernel<<<grid_filter, block, 0, stream>>>(
            reinterpret_cast<const float *>(dets),
            reinterpret_cast<const float *>(labels),
            total_images, num_queries,
            input_width, input_height,
            conf_thresh,
            d_coco_id_to_index,
            d_candidates, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());
    // 3. 一次性对所有 batch 按 score 降序排序（CUB 分段排序）
    int sort_total = total_images * num_queries;
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
    // 将排序后的候选框复制回 d_candidates，供 top-k 使用
    checkRuntime(cudaMemcpyAsync(
        d_candidates, d_sort_candidates_out,
        sort_total * sizeof(Candidate),
        cudaMemcpyDeviceToDevice, stream));
    // 4. 清零输出缓冲区 + mask buffer 用硬件加速清零
    const int total_out = total_images * max_detections;
    const int grid_out = (total_out + block - 1) / block;
    if (parse_masks && d_detection_masks != nullptr)
    {
        size_t mask_total = static_cast<size_t>(total_images) * max_detections * mask_output_resolution * mask_output_resolution;
        checkRuntime(cudaMemsetAsync(d_detection_masks, 0, mask_total * sizeof(float), stream));
    }
    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes,
        d_det_to_query_idx, d_mask_offsets, d_mask_shapes);
    checkRuntime(cudaPeekAtLastError());
    // 5. 写前 max_detections 个结果
    const int grid_batch = (total_images + block - 1) / block;
    write_topk_kernel<<<grid_batch, block, 0, stream>>>(
        d_candidates, d_counts,
        total_images, num_queries, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes,
        d_det_to_query_idx);
    checkRuntime(cudaPeekAtLastError());
    // 6. 计算 mask（可选）
    // float ms_masks = 0.0f;
    if (parse_masks && d_detection_masks != nullptr)
    {
        dim3 grid_masks(max_detections, total_images);
        if (input_is_half)
        {
            compute_masks_kernel<<<grid_masks, 256, 0, stream>>>(
                reinterpret_cast<const half *>(masks),
                d_boxes, d_num_dets, d_det_to_query_idx,
                total_images, num_queries, max_detections,
                input_width, input_height,
                mask_height, mask_width, mask_output_resolution,
                d_detection_masks, d_mask_offsets, d_mask_shapes);
        }
        else
        {
            compute_masks_kernel<<<grid_masks, 256, 0, stream>>>(
                reinterpret_cast<const float *>(masks),
                d_boxes, d_num_dets, d_det_to_query_idx,
                total_images, num_queries, max_detections,
                input_width, input_height,
                mask_height, mask_width, mask_output_resolution,
                d_detection_masks, d_mask_offsets, d_mask_shapes);
        }
        checkRuntime(cudaPeekAtLastError());
    }
}

} // namespace rfdetr_seg_postprocess
