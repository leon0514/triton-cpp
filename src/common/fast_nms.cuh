/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * 公共 fast NMS kernel：每框一线程，按 (score DESC, index ASC) 确定性去重
 * 供 yolo11 / yolov5 / sahi_ensemble 等所有后处理共用
 */

#ifndef __COMMON_FAST_NMS_CUH__
#define __COMMON_FAST_NMS_CUH__

#include "common/iou.cuh"
#include <cuda_runtime.h>

namespace common_nms
{

// ------------------------------------------------------------------
// 单图 fast NMS：image_offsets[i] 为第 i 张图起始索引，counts[i] 为候选数
// ------------------------------------------------------------------
static __global__ void fast_nms_image_kernel(
    const float *__restrict__ boxes,
    const float *__restrict__ scores,
    const int   *__restrict__ classes,
    const int   *__restrict__ image_offsets,
    const int   *__restrict__ counts,
    int num_images, int box_dim, float iou_threshold,
    int *__restrict__ keep_flags)
{
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    // 线性扫描找到 pos 属于哪个 image
    int img = 0, acc = 0;
    for (; img < num_images; ++img) {
        int cnt = counts[img];
        if (pos < acc + cnt) break;
        acc += cnt;
    }
    if (img >= num_images) return;

    // int local_idx = pos - acc;  // 在 image 内的编号
    int img_start = image_offsets[img];
    int img_count = counts[img];
    float cur_score = scores[pos];
    int   cur_class = classes[pos];

    for (int i = 0; i < img_count; ++i) {
        int other = img_start + i;
        if (other == pos || classes[other] != cur_class) continue;

        float other_score = scores[other];
        if (other_score > cur_score || (other_score == cur_score && other < pos)) {
            float iou;
            if (box_dim == 5) {
                iou = common_iou::box_probiou(
                    boxes[pos*5+0], boxes[pos*5+1], boxes[pos*5+2], boxes[pos*5+3], boxes[pos*5+4],
                    boxes[other*5+0], boxes[other*5+1], boxes[other*5+2], boxes[other*5+3], boxes[other*5+4]);
            } else {
                iou = common_iou::box_iou(
                    boxes[pos*4+0], boxes[pos*4+1], boxes[pos*4+2], boxes[pos*4+3],
                    boxes[other*4+0], boxes[other*4+1], boxes[other*4+2], boxes[other*4+3]);
            }
            if (iou > iou_threshold) { keep_flags[pos] = 0; return; }
        }
    }
}

// ------------------------------------------------------------------
// 跨图（SAHI）fast NMS：所有候选在一个 flat 数组，按 class 分组
// ------------------------------------------------------------------
static __global__ void fast_nms_flat_kernel(
    const float *__restrict__ boxes,
    const float *__restrict__ scores,
    const int   *__restrict__ classes,
    int N, int box_dim, float iou_threshold,
    int *__restrict__ keep_flags)
{
    int pos = blockDim.x * blockIdx.x + threadIdx.x;
    if (pos >= N) return;

    float cur_score = scores[pos];
    int   cur_class = classes[pos];

    for (int i = 0; i < N; ++i) {
        if (i == pos || classes[i] != cur_class) continue;
        float item_score = scores[i];
        if (item_score > cur_score || (item_score == cur_score && i < pos)) {
            float iou;
            if (box_dim == 5) {
                iou = common_iou::box_probiou(
                    boxes[pos*5+0], boxes[pos*5+1], boxes[pos*5+2], boxes[pos*5+3], boxes[pos*5+4],
                    boxes[i*5+0],   boxes[i*5+1],   boxes[i*5+2],   boxes[i*5+3],   boxes[i*5+4]);
            } else {
                iou = common_iou::box_iou(
                    boxes[pos*4+0], boxes[pos*4+1], boxes[pos*4+2], boxes[pos*4+3],
                    boxes[i*4+0],   boxes[i*4+1],   boxes[i*4+2],   boxes[i*4+3]);
            }
            if (iou > iou_threshold) { keep_flags[pos] = 0; return; }
        }
    }
}

} // namespace common_nms

#endif // __COMMON_FAST_NMS_CUH__
