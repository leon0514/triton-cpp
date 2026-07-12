/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo11_seg_postprocess/yolo11_seg_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace yolo11_seg_postprocess
{

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
    Candidate *candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * num_anchors;
    if (idx >= total)
        return;

    int batch_idx  = idx / num_anchors;
    int anchor_idx = idx % num_anchors;
    int num_channels = num_classes + 4 + num_masks;

    float cx = read_output0(input, batch_idx, anchor_idx, 0, num_anchors, num_channels, anchors_first);
    float cy = read_output0(input, batch_idx, anchor_idx, 1, num_anchors, num_channels, anchors_first);
    float w  = read_output0(input, batch_idx, anchor_idx, 2, num_anchors, num_channels, anchors_first);
    float h  = read_output0(input, batch_idx, anchor_idx, 3, num_anchors, num_channels, anchors_first);

    float x1 = cx - w * 0.5f;
    float y1 = cy - h * 0.5f;
    float x2 = cx + w * 0.5f;
    float y2 = cy + h * 0.5f;

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

    Candidate cand;
    cand.x1       = x1;
    cand.y1       = y1;
    cand.x2       = x2;
    cand.y2       = y2;
    cand.score    = score;
    cand.class_id = class_id;
    cand.batch_idx = batch_idx;
    cand.anchor_idx = anchor_idx;

    candidates[batch_idx * max_candidates + pos] = cand;
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

    int count = counts[b];
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

// 为每个保留检测框生成完整 prototype-resolution mask。
// 输出布局: [batch, max_detections, proto_h, proto_w]
template <typename T>
__global__ void compute_masks_kernel(
    const T *input,
    const T *protos,
    const int *num_dets,
    const int *det_to_cand_idx,
    const Candidate *candidates,
    int total_images,
    int num_anchors,
    int num_classes,
    int num_masks,
    int proto_h,
    int proto_w,
    bool anchors_first,
    int max_detections,
    int max_candidates,
    float *full_masks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * max_detections * proto_h * proto_w;
    if (idx >= total)
        return;

    int pw = idx % proto_w;
    int ph = (idx / proto_w) % proto_h;
    int det = (idx / (proto_h * proto_w)) % max_detections;
    int b = idx / (max_detections * proto_h * proto_w);

    int nd = num_dets[b];
    if (det >= nd)
    {
        full_masks[idx] = 0.0f;
        return;
    }

    int cand_idx = det_to_cand_idx[b * max_detections + det];
    const Candidate *cand = candidates + b * max_candidates;
    int anchor_idx = cand[cand_idx].anchor_idx;

    int num_channels = num_classes + 4 + num_masks;
    float val = 0.0f;
    for (int m = 0; m < num_masks; ++m)
    {
        float coeff = read_output0(
            input, b, anchor_idx, 4 + num_classes + m,
            num_anchors, num_channels, anchors_first);
        float proto_val = read_proto(
            protos, b, m, ph, pw,
            num_masks, proto_h, proto_w);
        val += coeff * proto_val;
    }

    // sigmoid
    float mask_val = 1.0f / (1.0f + expf(-val));
    full_masks[idx] = mask_val;
}

// 将完整 mask 裁剪到检测框（proto 坐标系），并按行优先展开拼接到 1D buffer。
__global__ void crop_and_flatten_masks_kernel(
    const float *full_masks,
    const int *num_dets,
    const float *boxes,
    int total_images,
    int max_detections,
    int proto_h,
    int proto_w,
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
    if (det >= nd)
    {
        mask_offsets[b * max_detections + det] = -1;
        mask_shapes[(b * max_detections + det) * 2 + 0] = 0;
        mask_shapes[(b * max_detections + det) * 2 + 1] = 0;
        return;
    }

    const float *box = boxes + (b * max_detections + det) * 4;

    // 将检测框（输入图像坐标）映射到 prototype 分辨率
    float scale_x = static_cast<float>(proto_w) / static_cast<float>(input_width);
    float scale_y = static_cast<float>(proto_h) / static_cast<float>(input_height);

    float x1p = box[0] * scale_x;
    float y1p = box[1] * scale_y;
    float x2p = box[2] * scale_x;
    float y2p = box[3] * scale_y;

    // 裁剪到原型边界
    int x1 = max(0, min(proto_w - 1, (int)floorf(x1p)));
    int y1 = max(0, min(proto_h - 1, (int)floorf(y1p)));
    int x2 = max(0, min(proto_w, (int)ceilf(x2p)));
    int y2 = max(0, min(proto_h, (int)ceilf(y2p)));

    int crop_w = x2 - x1;
    int crop_h = y2 - y1;

    int out_idx = b * max_detections * proto_h * proto_w + det * proto_h * proto_w;

    mask_offsets[b * max_detections + det] = out_idx;
    mask_shapes[(b * max_detections + det) * 2 + 0] = crop_h;
    mask_shapes[(b * max_detections + det) * 2 + 1] = crop_w;

    const float *src = full_masks + b * max_detections * proto_h * proto_w + det * proto_h * proto_w;
    float *dst = detection_masks + out_idx;

    // 每个线程处理 crop 区域一个像素
    int crop_size = crop_h * crop_w;
    for (int tid = threadIdx.x; tid < crop_size; tid += blockDim.x)
    {
        int dy = tid / crop_w;
        int dx = tid % crop_w;
        int sy = y1 + dy;
        int sx = x1 + dx;
        dst[tid] = src[sy * proto_w + sx];
    }
}

__global__ void init_candidates_kernel(
    int total_images,
    int max_candidates,
    Candidate *candidates)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_images * max_candidates;
    if (idx >= total)
        return;

    candidates[idx].score    = -FLT_MAX;
    candidates[idx].class_id = -1;
    candidates[idx].batch_idx = idx / max_candidates;
    candidates[idx].anchor_idx = 0;
}

__global__ void cap_counts_kernel(
    int total_images,
    int max_candidates,
    int *counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_images)
        return;

    if (counts[idx] > max_candidates)
        counts[idx] = max_candidates;
}

__global__ void init_output_kernel(
    int total_images,
    int max_detections,
    int proto_h,
    int proto_w,
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

    int mask_total = total_images * max_detections * proto_h * proto_w;
    for (int i = idx; i < mask_total; i += total)
    {
        detection_masks[i] = 0.0f;
    }

    if (idx < total_images)
        num_dets[idx] = 0;
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
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_detection_masks,
    int *d_mask_offsets,
    int *d_mask_shapes,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_anchors <= 0)
        return;

    // 初始化候选计数、候选框缓冲区和输出缓冲区
    checkRuntime(cudaMemsetAsync(d_counts, 0, total_images * sizeof(int), stream));

    int total_cand = total_images * max_candidates;
    int block_init = 256;
    int grid_cand  = (total_cand + block_init - 1) / block_init;
    init_candidates_kernel<<<grid_cand, block_init, 0, stream>>>(
        total_images, max_candidates, d_candidates);
    checkRuntime(cudaPeekAtLastError());

    int total_out = total_images * max_detections;
    int grid_out  = (total_out + block_init - 1) / block_init;
    init_output_kernel<<<grid_out, block_init, 0, stream>>>(
        total_images, max_detections, proto_h, proto_w,
        d_num_dets, d_boxes, d_scores, d_classes,
        d_detection_masks, d_mask_offsets, d_mask_shapes);
    checkRuntime(cudaPeekAtLastError());

    // decode + 置信度过滤
    int total_threads = total_images * num_anchors;
    int block = 256;
    int grid  = (total_threads + block - 1) / block;

    if (input_is_half)
    {
        decode_filter_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __half *>(input),
            total_images, num_anchors, num_classes, num_masks,
            anchors_first, apply_sigmoid, conf_thresh, max_candidates,
            d_candidates, d_counts);
    }
    else
    {
        decode_filter_kernel<<<grid, block, 0, stream>>>(
            static_cast<const float *>(input),
            total_images, num_anchors, num_classes, num_masks,
            anchors_first, apply_sigmoid, conf_thresh, max_candidates,
            d_candidates, d_counts);
    }
    checkRuntime(cudaPeekAtLastError());

    cap_counts_kernel<<<(total_images + block - 1) / block, block, 0, stream>>>(
        total_images, max_candidates, d_counts);
    checkRuntime(cudaPeekAtLastError());

    // 将候选计数拷贝到主机内存，再按 batch 排序（避免在主机端直接解引用设备指针）
    std::vector<int> h_counts(total_images);
    checkRuntime(cudaMemcpyAsync(
        h_counts.data(), d_counts, total_images * sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    for (int b = 0; b < total_images; ++b)
    {
        int count = h_counts[b];
        if (count > 1)
        {
            thrust::device_ptr<Candidate> cand_ptr(d_candidates + b * max_candidates);
            thrust::sort(thrust::cuda::par.on(stream), cand_ptr, cand_ptr + count, CandidateScoreGreater());
        }
    }

    // NMS
    int *d_det_to_cand_idx = nullptr;
    checkRuntime(cudaMallocAsync(
        &d_det_to_cand_idx, total_images * max_detections * sizeof(int), stream));
    checkRuntime(cudaMemsetAsync(
        d_det_to_cand_idx, 0, total_images * max_detections * sizeof(int), stream));

    nms_kernel<<<(total_images + block - 1) / block, block, 0, stream>>>(
        d_candidates, d_counts, total_images, max_candidates, max_detections,
        iou_thresh, d_num_dets, d_boxes, d_scores, d_classes, d_det_to_cand_idx);
    checkRuntime(cudaPeekAtLastError());

    // 计算完整 masks
    size_t full_mask_bytes = static_cast<size_t>(total_images) * max_detections * proto_h * proto_w * sizeof(float);
    float *d_full_masks = nullptr;
    checkRuntime(cudaMallocAsync(&d_full_masks, full_mask_bytes, stream));
    checkRuntime(cudaMemsetAsync(d_full_masks, 0, full_mask_bytes, stream));

    int total_mask_pixels = total_images * max_detections * proto_h * proto_w;
    int grid_mask = (total_mask_pixels + block - 1) / block;
    if (input_is_half)
    {
        compute_masks_kernel<<<grid_mask, block, 0, stream>>>(
            static_cast<const __half *>(input),
            static_cast<const __half *>(mask_protos),
            d_num_dets, d_det_to_cand_idx,
            d_candidates,
            total_images, num_anchors, num_classes, num_masks,
            proto_h, proto_w, anchors_first, max_detections,
            max_candidates,
            d_full_masks);
    }
    else
    {
        compute_masks_kernel<<<grid_mask, block, 0, stream>>>(
            static_cast<const float *>(input),
            static_cast<const float *>(mask_protos),
            d_num_dets, d_det_to_cand_idx,
            d_candidates,
            total_images, num_anchors, num_classes, num_masks,
            proto_h, proto_w, anchors_first, max_detections,
            max_candidates,
            d_full_masks);
    }
    checkRuntime(cudaPeekAtLastError());

    // 裁剪并展开到 1D buffer
    dim3 grid_crop(max_detections, total_images);
    crop_and_flatten_masks_kernel<<<grid_crop, 256, 0, stream>>>(
        d_full_masks, d_num_dets, d_boxes,
        total_images, max_detections, proto_h, proto_w,
        input_width, input_height,
        d_detection_masks, d_mask_offsets, d_mask_shapes);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaFreeAsync(d_full_masks, stream));
    checkRuntime(cudaFreeAsync(d_det_to_cand_idx, stream));
}

} // namespace yolo11_seg_postprocess
