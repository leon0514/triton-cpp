/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "yolo26_seg_postprocess/yolo26_seg_postprocess_kernel.hpp"
#include "common/check.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <float.h>

#include <cub/cub.cuh>

namespace yolo26_seg_postprocess
{

constexpr int kMaskOutputSize = 160;

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

template <typename T>
__global__ void gather_coefficients_kernel(
    const T *input,
    const Candidate *candidates,
    const int *num_dets,
    int total_images,
    int num_predictions,
    int num_masks,
    int max_detections,
    float *coefficients)
{
    int b = blockIdx.y;
    int det = blockIdx.x;

    if (b >= total_images || det >= num_dets[b])
        return;

    int pred_idx = candidates[b * num_predictions + det].pred_idx;
    int num_fields = 6 + num_masks;

    float *dst = coefficients + (b * max_detections + det) * num_masks;

    for (int i = threadIdx.x; i < num_masks; i += blockDim.x)
    {
        dst[i] = read_input(
            input,
            (b * num_predictions + pred_idx) * num_fields + 6 + i);
    }
}

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

void yolo26_seg_compute_masks_gpu(
    const void *input,
    const void *mask_protos,
    bool input_is_half,
    int total_images,
    int num_predictions,
    int num_masks,
    int proto_h,
    int proto_w,
    int input_width,
    int input_height,
    const int *d_num_dets,
    const float *d_boxes,
    const Candidate *d_candidates,
    int max_detections,
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

    checkRuntime(cudaMemcpyAsync(h_num_dets, d_num_dets, total_images * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    dim3 grid_coeffs(max_detections, total_images);
    if (input_is_half)
    {
        gather_coefficients_kernel<<<grid_coeffs, 256, 0, stream>>>(
            static_cast<const __half *>(input),
            d_candidates, d_num_dets,
            total_images, num_predictions, num_masks, max_detections,
            d_coefficients);
    }
    else
    {
        gather_coefficients_kernel<<<grid_coeffs, 256, 0, stream>>>(
            static_cast<const float *>(input),
            d_candidates, d_num_dets,
            total_images, num_predictions, num_masks, max_detections,
            d_coefficients);
    }
    checkRuntime(cudaPeekAtLastError());

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
            fprintf(stderr, "[yolo26_seg_compute_masks_gpu] cublasSgemm failed for batch %d: %d\n", b, status);
        }
    }

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
    init_output_kernel<<<grid_out, block, 0, stream>>>(
        total_images, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes,
        d_detection_masks, d_mask_offsets, d_mask_shapes);
    checkRuntime(cudaPeekAtLastError());

    const int grid_batch = (total_images + block - 1) / block;
    write_topk_kernel<<<grid_batch, block, 0, stream>>>(
        d_candidates, d_counts,
        total_images, num_predictions, max_detections,
        d_num_dets, d_boxes, d_scores, d_classes);
    checkRuntime(cudaPeekAtLastError());

    // mask 计算已拆分到 yolo26_seg_compute_masks_gpu 中，使用 cuBLAS GEMM + 轻量裁剪核。
}

} // namespace yolo26_seg_postprocess
