/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __YOLO11_OBB_ROTATED_IOU_CUH__
#define __YOLO11_OBB_ROTATED_IOU_CUH__

#include <cmath>

namespace yolo11_obb_postprocess
{

// ------------------------------------------------------------------
// 旋转框四个角点
// corners[8] = {x0,y0,x1,y1,x2,y2,x3,y3}，逆时针顺序
// ------------------------------------------------------------------
__host__ __device__ __forceinline__ void get_rotated_corners(
    float cx, float cy, float w, float h, float angle,
    float corners[8])
{
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float dx1 = w * 0.5f;
    float dy1 = h * 0.5f;

    // 未旋转时的半宽高
    float x[4] = {-dx1, dx1, dx1, -dx1};
    float y[4] = {-dy1, -dy1, dy1, dy1};

    for (int i = 0; i < 4; ++i)
    {
        float rx = x[i] * cos_a - y[i] * sin_a + cx;
        float ry = x[i] * sin_a + y[i] * cos_a + cy;
        corners[i * 2]     = rx;
        corners[i * 2 + 1] = ry;
    }
}

// ------------------------------------------------------------------
// 多边形面积（Shoelace 公式）
// poly: 顶点坐标数组 [x0,y0,x1,y1,...]
// n:   顶点数量
// ------------------------------------------------------------------
__host__ __device__ __forceinline__ float polygon_area(const float *poly, int n)
{
    if (n < 3)
        return 0.0f;

    float area = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        int j = (i + 1) % n;
        area += poly[i * 2] * poly[j * 2 + 1] - poly[j * 2] * poly[i * 2 + 1];
    }
    return fabsf(area) * 0.5f;
}

// ------------------------------------------------------------------
// Sutherland-Hodgman 多边形裁剪
// 用凸多边形 clipper 裁剪 subject，输出交多边形
// 两个旋转矩形均为凸四边形，因此最多产生 8 个交点
// ------------------------------------------------------------------
__host__ __device__ __forceinline__ void polygon_clip(
    const float *subject,
    int n_subject,
    const float *clipper,
    int n_clipper,
    float *output,
    int &n_output)
{
    // 当前多边形顶点缓冲区，最大 8 个顶点
    float current[16];
    int n_current = n_subject;
    for (int i = 0; i < n_subject; ++i)
    {
        current[i * 2]     = subject[i * 2];
        current[i * 2 + 1] = subject[i * 2 + 1];
    }

    for (int e = 0; e < n_clipper; ++e)
    {
        float ax = clipper[e * 2];
        float ay = clipper[e * 2 + 1];
        float bx = clipper[((e + 1) % n_clipper) * 2];
        float by = clipper[((e + 1) % n_clipper) * 2 + 1];
        float ex = bx - ax;
        float ey = by - ay;

        float next[16];
        int n_next = 0;

        for (int i = 0; i < n_current; ++i)
        {
            float px = current[i * 2];
            float py = current[i * 2 + 1];
            float qx = current[((i + 1) % n_current) * 2];
            float qy = current[((i + 1) % n_current) * 2 + 1];

            // 点相对于有向边 AB 的叉积，>0 表示在左侧（内侧）
            float cp = ex * (py - ay) - ey * (px - ax);
            float cq = ex * (qy - ay) - ey * (qx - ax);
            bool inside_p = cp >= 0.0f;
            bool inside_q = cq >= 0.0f;

            if (inside_p && inside_q)
            {
                // 都在内侧：保留终点 Q
                next[n_next * 2]     = qx;
                next[n_next * 2 + 1] = qy;
                ++n_next;
            }
            else if (inside_p && !inside_q)
            {
                // P 在内侧，Q 在外侧：保留交点
                float denom = cp - cq;
                float t = (fabsf(denom) > 1e-10f) ? cp / denom : 0.0f;
                next[n_next * 2]     = px + t * (qx - px);
                next[n_next * 2 + 1] = py + t * (qy - py);
                ++n_next;
            }
            else if (!inside_p && inside_q)
            {
                // P 在外侧，Q 在内侧：保留交点和终点 Q
                float denom = cp - cq;
                float t = (fabsf(denom) > 1e-10f) ? cp / denom : 0.0f;
                next[n_next * 2]     = px + t * (qx - px);
                next[n_next * 2 + 1] = py + t * (qy - py);
                ++n_next;
                next[n_next * 2]     = qx;
                next[n_next * 2 + 1] = qy;
                ++n_next;
            }
            // 都在外侧：不保留任何点
        }

        if (n_next == 0)
        {
            n_output = 0;
            return;
        }

        n_current = n_next;
        for (int i = 0; i < n_current; ++i)
        {
            current[i * 2]     = next[i * 2];
            current[i * 2 + 1] = next[i * 2 + 1];
        }
    }

    n_output = n_current;
    for (int i = 0; i < n_current; ++i)
    {
        output[i * 2]     = current[i * 2];
        output[i * 2 + 1] = current[i * 2 + 1];
    }
}

// ------------------------------------------------------------------
// 旋转矩形 IoU（Skew IoU / Rotated IoU）
// 基于多边形相交裁剪，比外接矩形 AABB IoU 更精确
// ------------------------------------------------------------------
__host__ __device__ __forceinline__ float rotated_iou(
    float cx1, float cy1, float w1, float h1, float angle1,
    float cx2, float cy2, float w2, float h2, float angle2)
{
    float corners1[8];
    float corners2[8];
    get_rotated_corners(cx1, cy1, w1, h1, angle1, corners1);
    get_rotated_corners(cx2, cy2, w2, h2, angle2, corners2);

    float intersection[16];
    int n_inter = 0;
    polygon_clip(corners1, 4, corners2, 4, intersection, n_inter);

    float inter_area = polygon_area(intersection, n_inter);
    float area1      = w1 * h1;
    float area2      = w2 * h2;
    float union_area = area1 + area2 - inter_area;

    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

} // namespace yolo11_obb_postprocess

#endif // __YOLO11_OBB_ROTATED_IOU_CUH__
