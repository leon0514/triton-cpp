/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "rfdetr_postprocess/rfdetr_postprocess_impl.hpp"
#include "common/check.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace rfdetr_postprocess
{

RfDetrPostprocess::RfDetrPostprocess(const RfDetrPostprocessConfig &config)
    : config_(config)
{
    // 根据 max_batch_size 预分配输出 workspace，避免推理热路径上的同步显存申请。
    const int max_batch = config_.max_batch_size;

    // 构造 COCO ID -> names 文件索引 的映射表，-1 表示跳过
    int host_coco_id_to_index[91];
    for (int i = 0; i <= 90; ++i)
        host_coco_id_to_index[i] = -1;

    int index = 0;
    for (int coco_id = 1; coco_id <= 90; ++coco_id)
    {
        if (std::find(config_.skip_coco_ids.begin(), config_.skip_coco_ids.end(), coco_id) !=
            config_.skip_coco_ids.end())
        {
            continue;
        }
        host_coco_id_to_index[coco_id] = index++;
    }

    coco_id_to_index_workspace_.gpu(91);
    checkRuntime(cudaMemcpy(coco_id_to_index_workspace_.gpu(), host_coco_id_to_index,
                            91 * sizeof(int), cudaMemcpyHostToDevice));

    num_detections_workspace_.gpu(max_batch);
    boxes_workspace_.gpu(max_batch * config_.max_detections * 4);
    scores_workspace_.gpu(max_batch * config_.max_detections);
    classes_workspace_.gpu(max_batch * config_.max_detections);

    // 中间缓冲区也根据 max_batch 与 num_queries 预分配
    counts_memory_.gpu(max_batch);
    candidates_memory_.gpu(max_batch * config_.num_queries);

    // CUB 分段排序工作区：一次性申请，推理时复用
    const int num_queries = config_.num_queries;
    sort_keys_in_workspace_.gpu(max_batch * num_queries);
    sort_keys_out_workspace_.gpu(max_batch * num_queries);
    sort_candidates_in_workspace_.gpu(max_batch * num_queries);
    sort_candidates_out_workspace_.gpu(max_batch * num_queries);
    sort_offsets_workspace_.gpu(max_batch + 1);

    // 预填分段偏移：每段长度为 num_queries
    std::vector<int> h_offsets(max_batch + 1);
    for (int i = 0; i <= max_batch; ++i)
    {
        h_offsets[i] = i * num_queries;
    }
    checkRuntime(cudaMemcpy(sort_offsets_workspace_.gpu(), h_offsets.data(),
                            (max_batch + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // 查询 CUB 临时存储大小（函数实现在 .cu 中，避免在 .cpp 中 include cub）
    cub_sort_temp_storage_bytes_ = get_segmented_sort_temp_storage_bytes(
        max_batch * num_queries, max_batch);
    if (cub_sort_temp_storage_bytes_ > 0)
    {
        cub_sort_temp_storage_workspace_.gpu(cub_sort_temp_storage_bytes_);
    }
}

void RfDetrPostprocess::forward(
    const void *dets,
    const void *labels,
    bool input_is_half,
    int total_images,
    int num_queries,
    cudaStream_t stream)
{
    if (total_images <= 0 || num_queries <= 0)
        return;

    int *d_counts      = counts_memory_.gpu(total_images);
    Candidate *d_cands = candidates_memory_.gpu(total_images * num_queries);

    int *d_num_dets = num_detections_workspace_.gpu(total_images);
    float *d_boxes  = boxes_workspace_.gpu(total_images * config_.max_detections * 4);
    float *d_scores = scores_workspace_.gpu(total_images * config_.max_detections);
    int *d_classes  = classes_workspace_.gpu(total_images * config_.max_detections);

    float *d_sort_keys_in = sort_keys_in_workspace_.gpu(total_images * num_queries);
    float *d_sort_keys_out = sort_keys_out_workspace_.gpu(total_images * num_queries);
    Candidate *d_sort_candidates_in = sort_candidates_in_workspace_.gpu(total_images * num_queries);
    Candidate *d_sort_candidates_out = sort_candidates_out_workspace_.gpu(total_images * num_queries);
    int *d_sort_offsets = sort_offsets_workspace_.gpu(total_images + 1);
    uint8_t *d_cub_temp = cub_sort_temp_storage_bytes_ > 0
                              ? cub_sort_temp_storage_workspace_.gpu()
                              : nullptr;

    // 若运行时的 num_queries 与构造时配置不同，重新生成偏移表
    if (num_queries != config_.num_queries)
    {
        std::vector<int> h_offsets(total_images + 1);
        for (int i = 0; i <= total_images; ++i)
        {
            h_offsets[i] = i * num_queries;
        }
        checkRuntime(cudaMemcpy(d_sort_offsets, h_offsets.data(),
                                (total_images + 1) * sizeof(int),
                                cudaMemcpyHostToDevice));
    }

    fprintf(stderr, "[rfdetr_postprocess] conf_thresh=%f input_size=%.0fx%.0f max_detections=%d\n",
            config_.confidence_threshold, config_.input_width, config_.input_height, config_.max_detections);
    fflush(stderr);

    rfdetr_postprocess_gpu(
        dets,
        labels,
        input_is_half,
        total_images,
        num_queries,
        config_.input_width,
        config_.input_height,
        config_.confidence_threshold,
        config_.max_detections,
        d_counts,
        d_cands,
        d_num_dets,
        d_boxes,
        d_scores,
        d_classes,
        coco_id_to_index_workspace_.gpu(),
        d_sort_keys_in,
        d_sort_keys_out,
        d_sort_candidates_in,
        d_sort_candidates_out,
        d_sort_offsets,
        d_cub_temp,
        cub_sort_temp_storage_bytes_,
        stream);
}

} // namespace rfdetr_postprocess
