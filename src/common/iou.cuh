/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * 公共 GPU IoU 函数：AABB IoU + OBB ProbIoU
 * 所有后处理（yolo11/obb/sahi_ensemble）共用
 */

#ifndef __COMMON_IOU_CUH__
#define __COMMON_IOU_CUH__

#include <cmath>

namespace common_iou
{

// ------------------------------------------------------------------
// AABB IoU
// ------------------------------------------------------------------
__host__ __device__ __forceinline__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom)
{
    float cleft   = fmaxf(aleft, bleft);
    float ctop    = fmaxf(atop, btop);
    float cright  = fminf(aright, bright);
    float cbottom = fminf(abottom, bbottom);
    float c_area  = fmaxf(cright - cleft, 0.0f) * fmaxf(cbottom - ctop, 0.0f);
    if (c_area == 0.0f) return 0.0f;
    float a_area = fmaxf(0.0f, aright - aleft) * fmaxf(0.0f, abottom - atop);
    float b_area = fmaxf(0.0f, bright - bleft) * fmaxf(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

// ------------------------------------------------------------------
// OBB ProbIoU（Gaussian 近似旋转 IoU）
// 参考: https://arxiv.org/pdf/2106.06072v1.pdf
// ------------------------------------------------------------------
__host__ __device__ __forceinline__ void covariance_matrix(
    float w, float h, float r, float& a, float& b, float& c)
{
    float a_val = w * w / 12.0f;
    float b_val = h * h / 12.0f;
    float cos_r = cosf(r), sin_r = sinf(r);
    a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
    b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
    c = (a_val - b_val) * sin_r * cos_r;
}

__host__ __device__ __forceinline__ float box_probiou(
    float cx1, float cy1, float w1, float h1, float r1,
    float cx2, float cy2, float w2, float h2, float r2, float eps = 1e-7f)
{
    float a1, b1, c1, a2, b2, c2;
    covariance_matrix(w1, h1, r1, a1, b1, c1);
    covariance_matrix(w2, h2, r2, a2, b2, c2);
    float t1 = ((a1+a2)*powf(cy1-cy2,2) + (b1+b2)*powf(cx1-cx2,2)) / ((a1+a2)*(b1+b2) - powf(c1+c2,2) + eps);
    float t2 = ((c1+c2)*(cx2-cx1)*(cy1-cy2)) / ((a1+a2)*(b1+b2) - powf(c1+c2,2) + eps);
    float t3 = logf(((a1+a2)*(b1+b2) - powf(c1+c2,2)) /
               (4.f*sqrtf(fmaxf(a1*b1-c1*c1,0.f))*sqrtf(fmaxf(a2*b2-c2*c2,0.f)) + eps) + eps);
    float bd = fmaxf(fminf(0.25f*t1 + 0.5f*t2 + 0.5f*t3, 100.0f), eps);
    return 1.f - sqrtf(1.f - expf(-bd) + eps);
}

} // namespace common_iou

#endif // __COMMON_IOU_CUH__
