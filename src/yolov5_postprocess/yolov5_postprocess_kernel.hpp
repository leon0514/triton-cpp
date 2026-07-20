/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLOV5_POSTPROCESS_KERNEL_HPP__
#define __YOLOV5_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace yolov5_postprocess
{

// 单个候选框（32 字节对齐，CUB 排序搬运和结构体拷贝走对齐事务）
struct Candidate
{
    float x1       = 0.0f;
    float y1       = 0.0f;
    float x2       = 0.0f;
    float y2       = 0.0f;
    float score    = 0.0f;
    int class_id   = 0;
    int batch_idx  = 0;
    int _pad       = 0;
};

static_assert(sizeof(Candidate) == 32, "Candidate must stay 32-byte aligned");

/**
 * @brief 对 YOLOv5 模型原始输出执行 decode + confidence 过滤 + NMS。
 *
 * 支持两种输入排布：
 *   - channels_first: [batch, C, num_anchors]
 *   - anchors_first:  [batch, num_anchors, C]
 *
 * 其中 C 取决于 has_objectness：
 *   - has_objectness = true:  C = 5 + num_classes（cx, cy, w, h, obj_conf, cls_0...cls_C-1）
 *   - has_objectness = false: C = 4 + num_classes（与 YOLO11 相同）
 *
 * @param input            输入数据指针（device）
 * @param input_is_half    输入是否为 FP16（否则为 FP32）
 * @param total_images     总图像数（所有 request 的 batch_size 之和）
 * @param num_anchors      anchor 数量（如 8400）
 * @param num_classes      类别数（如 80）
 * @param anchors_first    true 表示输入排布为 [batch, anchors, C]
 * @param apply_sigmoid    是否对 objectness / class score 应用 sigmoid
 *                       （Ultralytics 默认导出已做 sigmoid 时填 false）
 * @param has_objectness   true 表示输出包含 objectness 分支（YOLOv5 经典格式）
 * @param conf_thresh      置信度阈值
 * @param iou_thresh       NMS IoU 阈值
 * @param max_detections   每张图最多保留的检测框数
 * @param max_candidates   每张图最多进入 NMS 的候选框数
 * @param d_counts         每张图过滤后的候选数（device int[total_images]，输出，
 *                         数值可能超过 max_candidates，使用时需自行 min 封顶）
 * @param d_num_dets       每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes          检测框输出缓冲区（device float[total_images * max_detections * 4]）
 * @param d_scores         分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes        类别输出缓冲区（device int[total_images * max_detections]）
 * @param d_sort_keys_in   CUB 排序输入 key 缓冲区（device float[total_images * max_candidates]，
 *                         由 decode kernel 直接写入）
 * @param d_sort_keys_out  CUB 排序输出 key 缓冲区（device float[total_images * max_candidates]）
 * @param d_sort_candidates_in  CUB 排序输入 value 缓冲区（device Candidate[total_images * max_candidates]，
 *                         由 decode kernel 直接写入）
 * @param d_sort_candidates_out CUB 排序输出 value 缓冲区（device Candidate[total_images * max_candidates]，
 *                         NMS 直接读取）
 * @param d_sort_begin_offsets 每段起始偏移（device int[total_images]，由本函数内 kernel 填写）
 * @param d_sort_end_offsets   每段结束偏移（device int[total_images]，由本函数内 kernel 填写，
 *                         取 begin + min(count, max_candidates)，段间空隙不参与排序）
 * @param d_cub_temp       CUB 临时存储（可为 nullptr 当大小为 0）
 * @param cub_temp_storage_bytes CUB 临时存储字节数
 * @param stream           CUDA 流
 */
// 查询 CUB DeviceSegmentedRadixSort 所需临时存储字节数
size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments);

void yolov5_postprocess_gpu(
    const void *input,
    bool input_is_half,
    int total_images,
    int num_anchors,
    int num_classes,
    bool anchors_first,
    bool apply_sigmoid,
    bool has_objectness,
    float conf_thresh,
    float iou_thresh,
    int max_detections,
    int max_candidates,
    int *d_counts,
    int *d_num_dets,
    float *d_boxes,
    float *d_scores,
    int *d_classes,
    float *d_sort_keys_in,
    float *d_sort_keys_out,
    Candidate *d_sort_candidates_in,
    Candidate *d_sort_candidates_out,
    int *d_sort_begin_offsets,
    int *d_sort_end_offsets,
    void *d_cub_temp,
    size_t cub_temp_storage_bytes,
    cudaStream_t stream);

} // namespace yolov5_postprocess

#endif // __YOLOV5_POSTPROCESS_KERNEL_HPP__
