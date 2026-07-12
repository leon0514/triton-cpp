/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#ifndef __CLASSIFIER_POSTPROCESS_IMPL_HPP__
#define __CLASSIFIER_POSTPROCESS_IMPL_HPP__

#include "classifier_postprocess/classifier_postprocess_kernel.hpp"
#include "common/memory.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace classifier_postprocess
{

// 分类后处理配置
struct ClassifierPostprocessConfig
{
    int num_classes    = 1000;
    int top_k          = 5;
    bool apply_softmax = true; // 模型输出是否为原始 logits，需要 softmax
    int max_batch_size = 16;
};

class ClassifierPostprocess
{
  public:
    explicit ClassifierPostprocess(const ClassifierPostprocessConfig &config);
    ~ClassifierPostprocess() = default;

    ClassifierPostprocess(const ClassifierPostprocess &) = delete;
    ClassifierPostprocess &operator=(const ClassifierPostprocess &) = delete;

    /**
     * @brief 在指定 CUDA 流上执行分类后处理。
     *
     * 输出缓冲区为实例初始化时预分配的 GPU workspace，调用者通过
     * classes_gpu() / scores_gpu() 获取结果指针并拷贝到 response buffer。
     *
     * @param input         模型输出 logits/probs 数据指针（device）
     * @param input_is_half 输入是否为 FP16（否则为 FP32）
     * @param total_images  总图像数（动态 batch 之和）
     * @param stream        CUDA 流
     */
    void forward(
        const void *input,
        bool input_is_half,
        int total_images,
        cudaStream_t stream);

    inline const ClassifierPostprocessConfig &config() const { return config_; }
    inline int top_k() const { return config_.top_k; }

    inline int *classes_gpu() const { return classes_workspace_.gpu(); }
    inline float *scores_gpu() const { return scores_workspace_.gpu(); }

  private:
    ClassifierPostprocessConfig config_;

    // softmax 后概率（仅作为排序 key 使用）
    tensor::Memory<float> probs_workspace_;
    tensor::Memory<int> sort_values_workspace_;
    tensor::Memory<float> sorted_keys_workspace_;
    tensor::Memory<int> sorted_values_workspace_;

    tensor::Memory<int> classes_workspace_;
    tensor::Memory<float> scores_workspace_;

    // CUB DeviceSegmentedRadixSort 工作区
    tensor::Memory<int> sort_offsets_workspace_;
    tensor::Memory<uint8_t> cub_temp_storage_workspace_;
    size_t cub_temp_storage_bytes_ = 0;
};

} // namespace classifier_postprocess

#endif // __CLASSIFIER_POSTPROCESS_IMPL_HPP__
