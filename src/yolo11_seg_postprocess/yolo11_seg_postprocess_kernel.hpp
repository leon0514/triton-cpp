/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_SEG_POSTPROCESS_KERNEL_HPP__
#define __YOLO11_SEG_POSTPROCESS_KERNEL_HPP__

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace yolo11_seg_postprocess
{

// 单个候选框（32 字节对齐，CUB 排序搬运和结构体拷贝走对齐事务），
// 保留对应 anchor 索引以便后续读取 mask 系数
struct Candidate
{
    float x1       = 0.0f;
    float y1       = 0.0f;
    float x2       = 0.0f;
    float y2       = 0.0f;
    float score    = 0.0f;
    int class_id   = 0;
    int batch_idx  = 0;
    int anchor_idx = 0;  // 对应 output0 中的 anchor 索引
};

static_assert(sizeof(Candidate) == 32, "Candidate must stay 32-byte aligned");

/**
 * @brief 对 YOLO11-seg 模型原始输出执行 decode + confidence 过滤 + NMS + mask 生成。
 *
 * 输入排布：
 *   - output0 (channels_first): [batch, 4+num_classes+num_masks, num_anchors]
 *   - output0 (anchors_first):  [batch, num_anchors, 4+num_classes+num_masks]
 *   - output1:                  [batch, num_masks, proto_h, proto_w]
 *
 * 输出 mask 说明：
 *   对每个保留的检测框，按 prototype 分辨率计算 mask 并裁剪到检测框（proto 坐标系），
 *   最后按行优先展开并顺序拼接到 detection_masks 中。每个 mask 在 1D buffer 中的起始
 *   偏移写入 detection_mask_offsets，裁剪后的形状写入 detection_mask_shapes。
 *   1D buffer 的总容量为 max_detections * proto_h * proto_w。
 *
 * @param input                 output0 数据指针（device）
 * @param mask_protos           output1 prototype mask 数据指针（device）
 * @param input_is_half         输入是否为 FP16（否则为 FP32）
 * @param total_images          总图像数（所有 request 的 batch_size 之和）
 * @param num_anchors           anchor 数量（如 8400）
 * @param num_classes           类别数（如 80）
 * @param num_masks             mask 系数数量（如 32）
 * @param proto_h               prototype mask 高度（如 160）
 * @param proto_w               prototype mask 宽度（如 160） * @param mask_output_resolution      输出 mask 尺寸（正方形，如 160） * @param input_width           模型输入宽度（用于把检测框映射到 proto 分辨率）
 * @param input_height          模型输入高度
 * @param anchors_first         true 表示 output0 排布为 [batch, anchors, channels]
 * @param apply_sigmoid         是否对 class score 再应用一次 sigmoid（模型已做则填 false）
 * @param conf_thresh           置信度阈值
 * @param iou_thresh            NMS IoU 阈值
 * @param max_detections        每张图最多保留的检测框数
 * @param max_candidates        每张图最多进入 NMS 的候选框数
 * @param d_counts              每张图过滤后的候选数（device int[total_images]，输出，
 *                              数值可能超过 max_candidates，使用时需自行 min 封顶）
 * @param d_num_dets            每张图最终检测数（device int[total_images]，输出）
 * @param d_boxes               检测框输出缓冲区（device float[total_images * max_detections * 4]）
 * @param d_scores              分数输出缓冲区（device float[total_images * max_detections]）
 * @param d_classes             类别输出缓冲区（device int[total_images * max_detections]）
 * @param d_detection_masks     1D 拼接的 mask 数据（device float[total_images * max_detections * proto_h * proto_w]）
 * @param d_mask_offsets        每张图每个检测框的 mask 在 1D buffer 中的起始偏移
 *                              （device int[total_images * max_detections]，-1 表示无 mask）
 * @param d_mask_shapes         每张图每个检测框的 mask 形状 (h, w)
 *                              （device int[total_images * max_detections * 2]，0 表示无 mask）
 * @param d_det_to_cand_idx     每个保留检测框对应的候选框索引（device int[total_images * max_detections]）
 * @param d_sort_keys_in        CUB 排序输入 key 缓冲区（device float[total_images * max_candidates]，
 *                              由 decode kernel 直接写入）
 * @param d_sort_keys_out       CUB 排序输出 key 缓冲区（device float[total_images * max_candidates]）
 * @param d_sort_candidates_in  CUB 排序输入 value 缓冲区（device Candidate[total_images * max_candidates]，
 *                              由 decode kernel 直接写入）
 * @param d_sort_candidates_out CUB 排序输出 value 缓冲区（device Candidate[total_images * max_candidates]，
 *                              NMS 与 mask 计算直接读取）
 * @param d_sort_begin_offsets  每段起始偏移（device int[total_images]，由本函数内 kernel 填写）
 * @param d_sort_end_offsets    每段结束偏移（device int[total_images]，由本函数内 kernel 填写，
 *                              取 begin + min(count, max_candidates)，段间空隙不参与排序）
 * @param d_cub_temp            CUB 临时存储（可为 nullptr 当大小为 0）
 * @param cub_temp_storage_bytes CUB 临时存储字节数
 * @param stream                CUDA 流
 */
// 查询 CUB DeviceSegmentedRadixSort 所需临时存储字节数
size_t get_segmented_sort_temp_storage_bytes(
    int total_candidates, int num_segments);

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
    int mask_output_resolution,
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
    cudaStream_t stream);

/**
 * @brief 融合 kernel：系数读取 + matmul + crop + sigmoid 一步完成。
 *
 * 在调用本函数前，必须先调用 yolo11_seg_postprocess_gpu 完成 decode + NMS，
 * 得到 d_num_dets、d_boxes、d_det_to_cand_idx 和 d_sort_candidates_out。
 *
 * 与 cuBLAS GEMM 方案不同，此函数仅在 crop 区域内逐像素计算
 * dot(mask_weights, proto[:,y,x])，无中间 raw_masks buffer，
 * 对中小尺寸检测框显著减少无效计算。
 */
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
    int mask_output_resolution,
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
    cudaStream_t stream);

} // namespace yolo11_seg_postprocess

#endif // __YOLO11_SEG_POSTPROCESS_KERNEL_HPP__
