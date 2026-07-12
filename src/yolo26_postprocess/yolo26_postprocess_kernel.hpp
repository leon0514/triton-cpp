/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO26_POSTPROCESS_KERNEL_HPP__
#define __YOLO26_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>

namespace yolo26_postprocess
{

// YOLO26 one-to-one (NMS-free) 检测头输出的单个候选框
struct Candidate
{
    float x1      = 0.0f;
    float y1      = 0.0f;
    float x2      = 0.0f;
    float y2      = 0.0f;
    float score   = 0.0f;
    int class_id  = 0;
    int batch_idx = 0;
};

/**
 * @brief 对 YOLO26 one-to-one 头输出做置信度过滤与 top-K 排序。
 *
 * 输入格式: [batch, num_predictions, 6]
 * 6 列含义（默认）: [x1, y1, x2, y2, score, class]
 *
 * @param input          模型输出数据指针（device）
 * @param input_is_half  输入是否为 FP16（否则为 FP32）
 * @param total_images   总图像数
 * @param num_predictions 每图预测框数量（YOLO26 通常为 300）
 * @param conf_thresh    置信度阈值
 * @param max_detections 每张图最多保留的检测框数
 * @param d_counts       每张图过滤后的候选数（device int[total_images]，中间缓冲区）
 * @param d_candidates   候选框缓冲区（device Candidate[total_images * num_predictions]）
 * @param d_num_dets     每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes        检测框输出缓冲区（device float[total_images * max_detections * 4]）
 * @param d_scores       分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes      类别输出缓冲区（device int[total_images * max_detections]）
 * @param d_sort_keys_in        CUB 排序输入 key 缓冲区（device float[total_images * num_predictions]）
 * @param d_sort_keys_out       CUB 排序输出 key 缓冲区（device float[total_images * num_predictions]）
 * @param d_sort_candidates_in  CUB 排序输入 value 缓冲区（device Candidate[total_images * num_predictions]）
 * @param d_sort_candidates_out CUB 排序输出 value 缓冲区（device Candidate[total_images * num_predictions]）
 * @param d_sort_offsets        分段排序偏移（device int[total_images + 1]）
 * @param d_cub_temp            CUB 临时存储（可为 nullptr 当大小为 0）
 * @param cub_temp_storage_bytes CUB 临时存储字节数
 * @param stream                CUDA 流
 */
// 查询 CUB DeviceSegmentedRadixSort 所需临时存储字节数
size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments);

void yolo26_postprocess_gpu(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_predictions,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream);

} // namespace yolo26_postprocess

#endif // __YOLO26_POSTPROCESS_KERNEL_HPP__
