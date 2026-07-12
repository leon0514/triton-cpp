/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __RFDETR_POSTPROCESS_KERNEL_HPP__
#define __RFDETR_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>

namespace rfdetr_postprocess
{

// RF-DETR 单个候选框
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
 * @brief 对 RF-DETR 模型输出做后处理。
 *
 * 输入:
 *   - dets:   [batch, num_queries, 4]  归一化 cxcywh 框
 *   - labels: [batch, num_queries, 91] 类别 logits（最后一维为背景）
 *
 * 处理:
 *   - 去掉 labels 最后一维背景
 *   - 对前 90 维做 sigmoid，取最大值作为置信度/类别
 *   - 跳过配置中的空 COCO ID（通过 d_coco_id_to_index 映射）
 *   - 将归一化 cxcywh 转换为模型输入坐标系下的 xyxy
 *   - 按置信度排序，保留前 max_detections 个
 *   - 输出 class_id 为 names 文件中的 0-based 索引
 *
 * @param dets           dets 数据指针（device）
 * @param labels         labels 数据指针（device）
 * @param input_is_half  输入是否为 FP16（否则为 FP32）
 * @param total_images   总图像数
 * @param num_queries    每图 query 数（RF-DETR 通常为 300）
 * @param input_width    模型输入宽度（用于反归一化）
 * @param input_height   模型输入高度（用于反归一化）
 * @param conf_thresh    置信度阈值
 * @param max_detections 每张图最多保留的检测框数
 * @param d_counts       每张图过滤后的候选数（device int[total_images]，中间缓冲区）
 * @param d_candidates   候选框缓冲区（device Candidate[total_images * num_queries]）
 * @param d_num_dets     每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes        检测框输出缓冲区（device float[total_images * max_detections * 4]）
 * @param d_scores       分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes           类别输出缓冲区（device int[total_images * max_detections]）
 * @param d_coco_id_to_index  COCO ID -> names 文件索引 映射，-1 表示跳过（device int[91]）
 * @param d_sort_keys_in      CUB 排序输入 key 缓冲区（device float[total_images * num_queries]）
 * @param d_sort_keys_out     CUB 排序输出 key 缓冲区（device float[total_images * num_queries]）
 * @param d_sort_candidates_in CUB 排序输入 value 缓冲区（device Candidate[total_images * num_queries]）
 * @param d_sort_candidates_out CUB 排序输出 value 缓冲区（device Candidate[total_images * num_queries]）
 * @param d_sort_offsets      分段排序偏移（device int[total_images + 1]）
 * @param d_cub_temp          CUB 临时存储（可为 nullptr 当大小为 0）
 * @param cub_temp_storage_bytes CUB 临时存储字节数
 * @param stream              CUDA 流
 */
// 查询 CUB DeviceSegmentedRadixSort 所需临时存储字节数
size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments);

void rfdetr_postprocess_gpu(
    const void *dets,
    const void *labels,
    bool input_is_half,
    int total_images,
    int num_queries,
    float input_width,
    float input_height,
    float conf_thresh,
    int max_detections,
    int *d_counts,
    Candidate *d_candidates,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    const int *d_coco_id_to_index,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream);

} // namespace rfdetr_postprocess

#endif // __RFDETR_POSTPROCESS_KERNEL_HPP__
