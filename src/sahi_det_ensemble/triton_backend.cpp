/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * SAHI + Detection Ensemble — Triton C++ Backend
 *
 * 流程（全部 GPU 加速）：
 *   1. 接收 raw_image [H,W,3]
 *   2. 同步调用 SAHI_PREPROCESS → sliced_images + slice_offsets
 *      【关键】SAHI response 在分块推理完成前不释放，确保 GPU 内存有效
 *   3. 分块同步调用检测模型 → num_dets,boxes,scores,classes
 *   4. CUDA 一体化 kernel：置信度过滤 + 偏移校正 + 裁剪
 *   5. CUDA 逐类 NMS
 *   6. Host 端 Top-K 排序 + 填充输出
 *   7. CompletionQueue 异步发送响应，同时释放 SAHI response
 */

#include "sahi_det_ensemble/sahi_det_ensemble_kernel.hpp"
#include "sahi_det_ensemble/triton_config.hpp"
#include "common/device.hpp"
#include "common/memory.hpp"

#include <triton/core/tritonbackend.h>
#include <triton/core/tritonserver.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define BACKEND_NAME "sahi_det_ensemble"
#define RETURN_IF_ERROR(X) do { TRITONSERVER_Error *e = (X); if (e) return e; } while(0)
#define RETURN_TRITON_ERROR(C, M) return TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code::TRITONSERVER_ERROR_##C, (M))

using sahi_det_ensemble_backend::EnsembleConfig;
using sahi_det_ensemble_backend::ParseEnsembleConfig;

namespace backend
{

// ====================================================================
//  同步推理辅助（TRITONSERVER_ServerInferAsync + promise）
// ====================================================================

struct SyncCtx {
    std::promise<TRITONSERVER_InferenceResponse*> promise;
};

static TRITONSERVER_Error *GpuAlloc(
    TRITONSERVER_ResponseAllocator *, const char *,
    size_t byte_size, TRITONSERVER_MemoryType, int64_t preferred_id,
    void *, void **buffer, void **,
    TRITONSERVER_MemoryType *actual, int64_t *actual_id)
{
    cudaError_t e = cudaMalloc(buffer, byte_size);
    if (e != cudaSuccess)
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(e));
    *actual = TRITONSERVER_MEMORY_GPU;
    *actual_id = preferred_id;
    return nullptr;
}

static TRITONSERVER_Error *GpuFree(
    TRITONSERVER_ResponseAllocator *, void *buffer,
    void *, size_t, TRITONSERVER_MemoryType, int64_t)
{
    cudaError_t e = cudaFree(buffer);
    if (e != cudaSuccess)
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, cudaGetErrorString(e));
    return nullptr;
}

static void SyncCallback(
    TRITONSERVER_InferenceResponse *response, uint32_t, void *userp)
{
    auto *ctx = static_cast<SyncCtx *>(userp);
    ctx->promise.set_value(response);
}

static TRITONSERVER_Error *SyncInfer(
    TRITONSERVER_Server *server,
    TRITONSERVER_InferenceRequest *request,
    TRITONSERVER_InferenceResponse **response,
    TRITONSERVER_ResponseAllocator *allocator)
{
    SyncCtx ctx;
    auto future = ctx.promise.get_future();

    RETURN_IF_ERROR(TRITONSERVER_InferenceRequestSetResponseCallback(
        request, allocator, nullptr, SyncCallback, &ctx));

    RETURN_IF_ERROR(TRITONSERVER_ServerInferAsync(server, request, nullptr));

    *response = future.get();
    return nullptr;
}

// 从 InferenceResponse 中按名称提取输出
static TRITONSERVER_Error *GetOutputByName(
    TRITONSERVER_InferenceResponse *resp, const char *name,
    const void **buf, size_t *byte_size, std::vector<int64_t> *shape)
{
    uint32_t nout;
    RETURN_IF_ERROR(TRITONSERVER_InferenceResponseOutputCount(resp, &nout));
    for (uint32_t i = 0; i < nout; ++i) {
        const char *cn; TRITONSERVER_DataType dt; const int64_t *sh; uint64_t nd;
        const void *b; size_t sz; TRITONSERVER_MemoryType mt; int64_t mid;
        void *userp;
        RETURN_IF_ERROR(TRITONSERVER_InferenceResponseOutput(
            resp, i, &cn, &dt, &sh, &nd, &b, &sz, &mt, &mid, &userp));
        if (std::string(cn) == name) {
            *buf = b;
            *byte_size = sz;
            if (shape) shape->assign(sh, sh + nd);
            return nullptr;
        }
    }
    RETURN_TRITON_ERROR(INTERNAL, (std::string("output '") + name + "' not found").c_str());
}

// ====================================================================
//  CompletionQueue（异步发送响应 + 释放 SAHI response）
// ====================================================================

struct CompletionTask;

class CompletionQueue {
public:
    CompletionQueue() = default;
    void SetDeviceId(int did) { device_id_ = did; }

    void Stop() {
        { std::lock_guard<std::mutex> lk(mu_); shutdown_ = true; }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    ~CompletionQueue() { Stop(); }

    void Push(CompletionTask task);

private:
    void EnsureStarted();
    void Run();

    int device_id_ = 0;
    std::mutex start_mu_, mu_;
    std::condition_variable cv_;
    std::queue<CompletionTask> q_;
    bool shutdown_ = false;
    std::thread worker_;
};

struct CompletionTask {
    std::vector<TRITONBACKEND_Response *> responses;
    // 保活 SAHI + 检测 response，确保 GPU 内存（d_sliced 等）在发送前不被回收
    std::vector<TRITONSERVER_InferenceResponse *> sahi_responses;
    std::vector<TRITONSERVER_InferenceResponse *> det_responses;
    cudaEvent_t event = nullptr;
};

void CompletionQueue::EnsureStarted() {
    std::lock_guard<std::mutex> lk(start_mu_);
    if (!worker_.joinable())
        worker_ = std::thread(&CompletionQueue::Run, this);
}

void CompletionQueue::Run() {
    AutoDevice ad(device_id_);
    while (true) {
        CompletionTask task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this]{ return shutdown_ || !q_.empty(); });
            if (shutdown_ && q_.empty()) return;
            task = std::move(q_.front()); q_.pop();
        }
        if (task.event) {
            cudaEventSynchronize(task.event);
            cudaEventDestroy(task.event);
        }
        // 【关键】先释放 SAHI response — 归还 d_sliced 内存给 Triton 池
        for (auto *r : task.sahi_responses)
            if (r) TRITONSERVER_InferenceResponseDelete(r);
        // 再释放检测 response
        for (auto *r : task.det_responses)
            if (r) TRITONSERVER_InferenceResponseDelete(r);

        for (auto *resp : task.responses) {
            if (resp)
                TRITONBACKEND_ResponseSend(resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
        }
    }
}

void CompletionQueue::Push(CompletionTask task) {
    EnsureStarted();
    { std::lock_guard<std::mutex> lk(mu_); q_.push(std::move(task)); }
    cv_.notify_one();
}

// ====================================================================
//  State
// ====================================================================

struct ModelState {
    TRITONBACKEND_Model *model = nullptr;
    TRITONSERVER_Server *server = nullptr;
    EnsembleConfig config;
    TRITONSERVER_ResponseAllocator *allocator = nullptr;

    explicit ModelState(TRITONBACKEND_Model *m) : model(m) {
        TRITONBACKEND_ModelServer(model, &server);
        TRITONSERVER_ResponseAllocatorNew(&allocator, GpuAlloc, GpuFree, nullptr);
    }

    ~ModelState() {
        if (allocator) TRITONSERVER_ResponseAllocatorDelete(allocator);
    }

    TRITONSERVER_Error *LoadConfig() {
        TRITONSERVER_Message *msg;
        RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(model, 1, &msg));
        auto err = ParseEnsembleConfig(msg, config);
        TRITONSERVER_MessageDelete(msg);
        return err;
    }
};

struct InstanceState {
    TRITONBACKEND_ModelInstance *instance = nullptr;
    int device_id = 0;
    cudaStream_t stream = nullptr;
    EnsembleConfig cfg;
    CompletionQueue completion_queue;

    // 检测模型输出的 GPU workspace
    tensor::Memory<int> det_num_dets_;
    tensor::Memory<float> det_boxes_;
    tensor::Memory<float> det_scores_;
    tensor::Memory<int> det_classes_;

    // 过滤/偏移后的结果
    tensor::Memory<float> filtered_boxes_;
    tensor::Memory<float> filtered_scores_;
    tensor::Memory<int> filtered_classes_;
    tensor::Memory<int> filtered_slice_idx_;
    tensor::Memory<int> filtered_count_;

    // NMS 工作区
    tensor::Memory<int> nms_keep_;
    tensor::Memory<int> nms_kept_;
    tensor::Memory<int> nms_offsets_;
    tensor::Memory<int> nms_counters_;
    tensor::Memory<int> nms_flags_;

    // SAHI 输出的 slice_offsets（GPU 上）
    tensor::Memory<int> slice_offsets_;

    explicit InstanceState(TRITONBACKEND_ModelInstance *inst) : instance(inst) {
        TRITONBACKEND_ModelInstanceDeviceId(inst, &device_id);
        completion_queue.SetDeviceId(device_id);
    }

    ~InstanceState() {
        completion_queue.Stop();
        if (stream) cudaStreamDestroy(stream);
    }

    TRITONSERVER_Error *Init(ModelState *ms) {
        AutoDevice ad(device_id);
        cfg = ms->config;

        cudaError_t ce = cudaStreamCreate(&stream);
        if (ce != cudaSuccess) RETURN_TRITON_ERROR(INTERNAL, cudaGetErrorString(ce));

        int max_slices = cfg.max_slices;
        int max_dets = cfg.max_detections;
        int max_total = max_slices * max_dets;

        det_num_dets_.gpu(max_slices);
        det_boxes_.gpu(max_slices * max_dets * 4);
        det_scores_.gpu(max_slices * max_dets);
        det_classes_.gpu(max_slices * max_dets);

        filtered_boxes_.gpu(max_total * 4);
        filtered_scores_.gpu(max_total);
        filtered_classes_.gpu(max_total);
        filtered_slice_idx_.gpu(max_total);
        filtered_count_.gpu(1);

        nms_keep_.gpu(max_total);
        nms_kept_.gpu(1);
        nms_offsets_.gpu(cfg.num_classes + 1);
        nms_counters_.gpu(cfg.num_classes);
        nms_flags_.gpu(max_total);

        slice_offsets_.gpu(max_slices * 4);

        return nullptr;
    }
};

// ====================================================================
//  请求处理
// ====================================================================

struct ReqInfo {
    TRITONBACKEND_Request *req = nullptr;
    TRITONBACKEND_Response *resp = nullptr;  // 置为 nullptr 表示已发送，防止双发
    int H = 0, W = 0;
    const uint8_t *img = nullptr;
    bool on_gpu = false;
    int64_t mem_id = 0;
    tensor::Memory<uint8_t> staging;

    // SAHI response — 保活 GPU 指针（d_sliced / d_slice_off），由 CompletionQueue 释放
    TRITONSERVER_InferenceResponse *sahi_resp = nullptr;
    uint8_t *d_sliced = nullptr;
    int *d_slice_off = nullptr;
    int slice_num = 0;

    // 检测 response 容器 — 分块推理期间保活，由 CompletionQueue 统一释放
    std::vector<TRITONSERVER_InferenceResponse *> det_resps;
};

static TRITONSERVER_Error *ParseRequest(TRITONBACKEND_Request *req, ReqInfo &info) {
    info.req = req;
    uint32_t nc; RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(req, &nc));
    if (nc != 1) RETURN_TRITON_ERROR(INVALID_ARG, "need 1 input");

    TRITONBACKEND_Input *inp;
    RETURN_IF_ERROR(TRITONBACKEND_RequestInputByIndex(req, 0, &inp));

    const char *name; TRITONSERVER_DataType dt; const int64_t *shape;
    uint32_t ndims, nbuf; uint64_t nbytes;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(inp, &name, &dt, &shape, &ndims, &nbytes, &nbuf));
    if (dt != TRITONSERVER_TYPE_UINT8) RETURN_TRITON_ERROR(INVALID_ARG, "need UINT8");
    if (nbuf != 1) RETURN_TRITON_ERROR(INVALID_ARG, "need 1 buffer");

    if (ndims == 4) { info.H = static_cast<int>(shape[1]); info.W = static_cast<int>(shape[2]); }
    else if (ndims == 3) { info.H = static_cast<int>(shape[0]); info.W = static_cast<int>(shape[1]); }
    else RETURN_TRITON_ERROR(INVALID_ARG, "need 3D or 4D");

    const void *buf; TRITONSERVER_MemoryType mt; int64_t mid;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(inp, 0, &buf, &nbytes, &mt, &mid));
    info.img = static_cast<const uint8_t *>(buf);
    info.on_gpu = (mt == TRITONSERVER_MEMORY_GPU);
    info.mem_id = mid;
    return nullptr;
}

} // namespace backend

// ====================================================================
//  Triton C API
// ====================================================================

extern "C" {

TRITONSERVER_Error *TRITONBACKEND_Initialize(TRITONBACKEND_Backend *b) {
    const char *n; RETURN_IF_ERROR(TRITONBACKEND_BackendName(b, &n));
    if (std::string(n) != BACKEND_NAME) RETURN_TRITON_ERROR(INTERNAL, "name mismatch");
    uint32_t maj, min; RETURN_IF_ERROR(TRITONBACKEND_ApiVersion(&maj, &min));
    if (maj != TRITONBACKEND_API_VERSION_MAJOR || min < TRITONBACKEND_API_VERSION_MINOR)
        RETURN_TRITON_ERROR(INTERNAL, "API version mismatch");
    return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *m) {
    auto s = std::make_unique<backend::ModelState>(m);
    RETURN_IF_ERROR(s->LoadConfig());
    TRITONBACKEND_ModelSetState(m, s.release());
    return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *m) {
    void *p; RETURN_IF_ERROR(TRITONBACKEND_ModelState(m, &p));
    delete static_cast<backend::ModelState *>(p);
    return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *inst) {
    TRITONBACKEND_Model *model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(inst, &model));
    backend::ModelState *ms;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, reinterpret_cast<void **>(&ms)));

    auto is = std::make_unique<backend::InstanceState>(inst);
    RETURN_IF_ERROR(is->Init(ms));
    TRITONBACKEND_ModelInstanceSetState(inst, is.release());
    return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *inst) {
    void *p; RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(inst, &p));
    delete static_cast<backend::InstanceState *>(p);
    return nullptr;
}

TRITONSERVER_Error *TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance *inst,
    TRITONBACKEND_Request **requests, uint32_t cnt)
{
    using namespace backend;

    InstanceState *is;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(inst, reinterpret_cast<void **>(&is)));

    TRITONBACKEND_Model *model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(inst, &model));
    ModelState *ms;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, reinterpret_cast<void **>(&ms)));

    int dev; RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(inst, &dev));
    AutoDevice ad(dev);

    cudaStream_t stream = is->stream;
    const auto &cfg = is->cfg;
    auto *server = ms->server;
    auto *allocator = ms->allocator;

    // ---- 1. 解析所有请求 ----
    std::vector<ReqInfo> infos(cnt);
    for (uint32_t i = 0; i < cnt; ++i) {
        auto err = ParseRequest(requests[i], infos[i]);
        if (err) {
            TRITONBACKEND_Response *r = nullptr;
            if (!TRITONBACKEND_ResponseNew(&r, requests[i]))
                TRITONBACKEND_ResponseSend(r, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
            TRITONSERVER_ErrorDelete(err);
        }
    }

    // ---- 2. 逐请求处理 ----
    for (auto &info : infos) {
        if (!info.req) continue;
        TRITONBACKEND_ResponseNew(&info.resp, info.req);

        // ---- 2a. 确保输入在 GPU 上 ----
        const uint8_t *d_img = info.img;
        if (!info.on_gpu || info.mem_id != dev) {
            d_img = info.staging.gpu(info.H * info.W * 3);
            cudaMemcpyAsync(const_cast<uint8_t *>(d_img), info.img,
                            info.H * info.W * 3, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        }

        // ---- 2b. 调用 SAHI_PREPROCESS（同步推理，保活 response） ----
        TRITONSERVER_InferenceRequest *sahi_req = nullptr;
        auto err = TRITONSERVER_InferenceRequestNew(
            &sahi_req, server, "SAHI_PREPROCESS", -1);
        if (!err) {
            const int64_t sh[3] = {info.H, info.W, 3};
            err = TRITONSERVER_InferenceRequestAddInput(
                sahi_req, "raw_image", TRITONSERVER_TYPE_UINT8, sh, 3);
        }
        if (!err)
            err = TRITONSERVER_InferenceRequestAppendInputData(
                sahi_req, "raw_image", d_img, info.H * info.W * 3,
                TRITONSERVER_MEMORY_GPU, dev);
        if (!err)
            err = TRITONSERVER_InferenceRequestAddRequestedOutput(sahi_req, "sliced_images");
        if (!err)
            err = TRITONSERVER_InferenceRequestAddRequestedOutput(sahi_req, "slice_offsets");

        if (err) {
            if (sahi_req) TRITONSERVER_InferenceRequestDelete(sahi_req);
            TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
            info.resp = nullptr;
            TRITONSERVER_ErrorDelete(err);
            continue;
        }

        TRITONSERVER_InferenceResponse *sahi_resp = nullptr;
        err = SyncInfer(server, sahi_req, &sahi_resp, allocator);
        TRITONSERVER_InferenceRequestDelete(sahi_req);

        if (err) {
            TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
            info.resp = nullptr;
            TRITONSERVER_ErrorDelete(err);
            continue;
        }

        err = TRITONSERVER_InferenceResponseError(sahi_resp);
        if (err) {
            TRITONSERVER_InferenceResponseDelete(sahi_resp);
            TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
            info.resp = nullptr;
            TRITONSERVER_ErrorDelete(err);
            continue;
        }

        // 从输出 shape 获取切片数
        const void *sliced_buf = nullptr, *off_buf = nullptr;
        size_t sliced_sz = 0, off_sz = 0;
        std::vector<int64_t> sliced_shape, off_shape;

        err = GetOutputByName(sahi_resp, "sliced_images", &sliced_buf, &sliced_sz, &sliced_shape);
        if (!err)
            err = GetOutputByName(sahi_resp, "slice_offsets", &off_buf, &off_sz, &off_shape);

        if (err) {
            TRITONSERVER_InferenceResponseDelete(sahi_resp);
            TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
            info.resp = nullptr;
            TRITONSERVER_ErrorDelete(err);
            continue;
        }

        int slice_num = (off_shape.size() >= 1) ? static_cast<int>(off_shape[0]) : 0;

        // ---- 【安全边界】slice_num > max_slices 防御 ----
        if (slice_num > cfg.max_slices) {
            TRITONSERVER_InferenceResponseDelete(sahi_resp);
            auto *bound_err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                "slice_num exceeds max_slices, increase max_slices in config.pbtxt");
            TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, bound_err);
            info.resp = nullptr;
            TRITONSERVER_ErrorDelete(bound_err);
            continue;
        }

        info.d_sliced = const_cast<uint8_t *>(static_cast<const uint8_t *>(sliced_buf));
        info.d_slice_off = const_cast<int *>(static_cast<const int *>(off_buf));
        info.slice_num = slice_num;
        info.sahi_resp = sahi_resp;  // 【保活】在 CompletionQueue 中释放

        // ---- 2c. 分块调用检测模型 ----
        // 【修改】不再在循环内释放 det_resp 或做同步；
        //        将 det_resp 保活到 info.det_resps，由 CompletionQueue 统一清理。
        size_t slice_px = static_cast<size_t>(cfg.slice_width) * cfg.slice_height * 3;
        bool det_ok = true;

        for (int s = 0; s < slice_num; s += cfg.chunk_size) {
            int e = std::min(s + cfg.chunk_size, slice_num);
            int n = e - s;

            TRITONSERVER_InferenceRequest *det_req = nullptr;
            err = TRITONSERVER_InferenceRequestNew(
                &det_req, server, cfg.detector_model.c_str(), -1);

            if (!err) {
                const int64_t det_sh[4] = {n, cfg.slice_height, cfg.slice_width, 3};
                size_t det_bytes = static_cast<size_t>(n) * slice_px;
                err = TRITONSERVER_InferenceRequestAddInput(
                    det_req, "raw_image", TRITONSERVER_TYPE_UINT8, det_sh, 4);
                if (!err)
                    err = TRITONSERVER_InferenceRequestAppendInputData(
                        det_req, "raw_image",
                        info.d_sliced + s * slice_px, det_bytes,
                        TRITONSERVER_MEMORY_GPU, dev);
            }
            if (!err) {
                const char *outs[] = {"num_dets", "detection_boxes",
                                      "detection_scores", "detection_classes"};
                for (auto *o : outs)
                    err = TRITONSERVER_InferenceRequestAddRequestedOutput(det_req, o);
            }

            if (err) {
                if (det_req) TRITONSERVER_InferenceRequestDelete(det_req);
                TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
                info.resp = nullptr;
                TRITONSERVER_ErrorDelete(err);
                det_ok = false; break;
            }

            TRITONSERVER_InferenceResponse *det_resp = nullptr;
            err = SyncInfer(server, det_req, &det_resp, allocator);
            TRITONSERVER_InferenceRequestDelete(det_req);

            if (err) {
                TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
                info.resp = nullptr;
                TRITONSERVER_ErrorDelete(err);
                det_ok = false; break;
            }

            err = TRITONSERVER_InferenceResponseError(det_resp);
            if (err) {
                TRITONSERVER_InferenceResponseDelete(det_resp);
                TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, err);
                info.resp = nullptr;
                TRITONSERVER_ErrorDelete(err);
                det_ok = false; break;
            }

            const void *nd_b = nullptr, *bx_b = nullptr, *sc_b = nullptr, *cl_b = nullptr;
            size_t nd_sz = 0, bx_sz = 0, sc_sz = 0, cl_sz = 0;
            std::vector<int64_t> nd_sh, bx_sh, sc_sh, cl_sh;

            auto e1 = GetOutputByName(det_resp, "num_dets", &nd_b, &nd_sz, &nd_sh);
            auto e2 = GetOutputByName(det_resp, "detection_boxes", &bx_b, &bx_sz, &bx_sh);
            auto e3 = GetOutputByName(det_resp, "detection_scores", &sc_b, &sc_sz, &sc_sh);
            auto e4 = GetOutputByName(det_resp, "detection_classes", &cl_b, &cl_sz, &cl_sh);

            if (e1 || e2 || e3 || e4) {
                if (e1) TRITONSERVER_ErrorDelete(e1);
                if (e2) TRITONSERVER_ErrorDelete(e2);
                if (e3) TRITONSERVER_ErrorDelete(e3);
                if (e4) TRITONSERVER_ErrorDelete(e4);
                TRITONSERVER_InferenceResponseDelete(det_resp);
                // 【修复】必须发送 Error Response，否则客户端无限挂起
                auto *parse_err = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                    "failed to parse detector chunk output");
                TRITONBACKEND_ResponseSend(info.resp, TRITONSERVER_RESPONSE_COMPLETE_FINAL, parse_err);
                TRITONSERVER_ErrorDelete(parse_err);
                info.resp = nullptr;
                det_ok = false; break;
            }

            // 【修复】检测器输出步长 = actual_num_dets，workspace 步长 = max_detections
            // 必须逐行 pitched copy，不能用 flat memcpy
            int actual_num_dets = (bx_sh.size() >= 2) ? static_cast<int>(bx_sh[1]) : cfg.max_detections;
            if (actual_num_dets <= 0) actual_num_dets = cfg.max_detections;

            // num_dets: [n] → flat copy 即可
            cudaMemcpyAsync(is->det_num_dets_.gpu() + s, nd_b,
                            std::min(nd_sz, static_cast<size_t>(n) * sizeof(int)),
                            cudaMemcpyDeviceToDevice, stream);

            // boxes: [n, actual_num_dets, 4] → strided copy to [n, max_dets, 4]
            sahi_det_ensemble::strided_copy_boxes(
                static_cast<const float *>(bx_b),
                is->det_boxes_.gpu() + s * cfg.max_detections * 4,
                n, actual_num_dets * 4, cfg.max_detections * 4, stream);

            // scores: [n, actual_num_dets] → strided copy to [n, max_dets]
            sahi_det_ensemble::strided_copy_scores(
                static_cast<const float *>(sc_b),
                is->det_scores_.gpu() + s * cfg.max_detections,
                n, actual_num_dets, cfg.max_detections, stream);

            // classes: [n, actual_num_dets] → strided copy to [n, max_dets]
            sahi_det_ensemble::strided_copy_classes(
                static_cast<const int *>(cl_b),
                is->det_classes_.gpu() + s * cfg.max_detections,
                n, actual_num_dets, cfg.max_detections, stream);

            // 【修改】延迟释放 — 保活 det_resp 到 info.det_resps
            info.det_resps.push_back(det_resp);
        }

        if (!det_ok) {
            // 【修改】清理已保活的 det_resp，防止内存泄漏
            for (auto *r : info.det_resps)
                if (r) TRITONSERVER_InferenceResponseDelete(r);
            info.det_resps.clear();
            continue;
        }

        // ---- 2d. 拷贝 slice_offsets 到 workspace ----
        cudaMemcpyAsync(is->slice_offsets_.gpu(), info.d_slice_off,
                        slice_num * 4 * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        // ---- 2e. filter_and_offset（一体化 CUDA kernel） ----
        int max_total = slice_num * cfg.max_detections;
        // ---- 2e. filter_and_offset（一体化 CUDA kernel） ----
        sahi_det_ensemble::filter_and_offset(
            is->det_num_dets_.gpu(), is->det_boxes_.gpu(),
            is->det_scores_.gpu(), is->det_classes_.gpu(),
            is->slice_offsets_.gpu(),
            slice_num, cfg.max_detections,
            cfg.confidence_threshold, info.W, info.H,
            is->filtered_boxes_.gpu(), is->filtered_scores_.gpu(),
            is->filtered_classes_.gpu(), is->filtered_slice_idx_.gpu(),
            is->filtered_count_.gpu(), max_total, stream);

        // ---- 【修改 1】同步获取精确的过滤后框数 h_n ----
        int h_n = 0;
        cudaMemcpyAsync(&h_n, is->filtered_count_.gpu(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // ---- 2f. 逐类 NMS（仅在有有效框时运行，传入精准的 h_n） ----
        int h_nms = 0;
        if (h_n > 0) {
            sahi_det_ensemble::nms_per_class(
                is->filtered_boxes_.gpu(), is->filtered_scores_.gpu(),
                is->filtered_classes_.gpu(),
                h_n,  // 【修改 2】恢复传入精准的 h_n，防止 GPU 越界
                cfg.iou_threshold, cfg.num_classes,
                is->nms_keep_.gpu(), is->nms_kept_.gpu(),
                is->nms_offsets_.gpu(), is->nms_counters_.gpu(),
                is->nms_flags_.gpu(), stream);

            // ---- 【修改 3】异步拷贝 NMS 保留框数并同步 ----
            cudaMemcpyAsync(&h_nms, is->nms_kept_.gpu(), sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        // ---- 2g. 批量 GPU→CPU 拷贝（仅拷贝有效区间，节省带宽） ----
        std::vector<int> h_keep(h_nms);
        std::vector<float> h_fb(h_n * 4), h_fs(h_n);
        std::vector<int> h_fc(h_n);

        if (h_nms > 0) {
            cudaMemcpyAsync(h_keep.data(), is->nms_keep_.gpu(),
                            h_nms * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_fb.data(), is->filtered_boxes_.gpu(),
                            h_n * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_fs.data(), is->filtered_scores_.gpu(),
                            h_n * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_fc.data(), is->filtered_classes_.gpu(),
                            h_n * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        // ---- 2h. 填充输出 ----
        auto alloc_output = [&](const char *oname, TRITONSERVER_DataType dt,
                                const int64_t *oshape, uint32_t odims,
                                size_t obytes, void **obuf) -> TRITONSERVER_Error * {
            TRITONBACKEND_Output *out;
            RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(info.resp, &out, oname, dt, oshape, odims));
            TRITONSERVER_MemoryType mt = TRITONSERVER_MEMORY_CPU;
            int64_t mid = 0;
            RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(out, obuf, obytes, &mt, &mid));
            return nullptr;
        };

        if (h_nms == 0) {
            const int64_t ns[1] = {1}, bs[2] = {cfg.max_detections, 4},
                          ss[1] = {cfg.max_detections}, cs[1] = {cfg.max_detections};
            int32_t zero = 0; void *nb = nullptr;
            alloc_output("num_dets", TRITONSERVER_TYPE_INT32, ns, 1, 4, &nb);
            if (nb) memcpy(nb, &zero, 4);
            void *bb = nullptr; alloc_output("detection_boxes", TRITONSERVER_TYPE_FP32,
                bs, 2, cfg.max_detections * 4 * sizeof(float), &bb);
            if (bb) memset(bb, 0, cfg.max_detections * 4 * sizeof(float));
            void *sb = nullptr; alloc_output("detection_scores", TRITONSERVER_TYPE_FP32,
                ss, 1, cfg.max_detections * sizeof(float), &sb);
            if (sb) memset(sb, 0, cfg.max_detections * sizeof(float));
            void *cb = nullptr; alloc_output("detection_classes", TRITONSERVER_TYPE_INT32,
                cs, 1, cfg.max_detections * sizeof(int32_t), &cb);
            if (cb) memset(cb, -1, cfg.max_detections * sizeof(int32_t));
            continue;
        }

        // 按分数排序
        std::vector<std::pair<float, int>> scored;
        for (int i = 0; i < h_nms; ++i)
            scored.emplace_back(-h_fs[h_keep[i]], i);
        std::sort(scored.begin(), scored.end());

        int final_n = std::min(h_nms, cfg.max_detections);

        const int64_t ns[1] = {1}, bs[2] = {cfg.max_detections, 4},
                      ss[1] = {cfg.max_detections}, cs[1] = {cfg.max_detections};

        int32_t nd_val = final_n; void *nb = nullptr;
        alloc_output("num_dets", TRITONSERVER_TYPE_INT32, ns, 1, 4, &nb);
        if (nb) memcpy(nb, &nd_val, 4);

        void *bb = nullptr;
        alloc_output("detection_boxes", TRITONSERVER_TYPE_FP32,
                     bs, 2, cfg.max_detections * 4 * sizeof(float), &bb);
        void *sb = nullptr;
        alloc_output("detection_scores", TRITONSERVER_TYPE_FP32,
                     ss, 1, cfg.max_detections * sizeof(float), &sb);
        void *cb = nullptr;
        alloc_output("detection_classes", TRITONSERVER_TYPE_INT32,
                     cs, 1, cfg.max_detections * sizeof(int32_t), &cb);

        if (bb) {
            float *fb = static_cast<float *>(bb);
            memset(fb, 0, cfg.max_detections * 4 * sizeof(float));
            for (int i = 0; i < final_n; ++i) {
                int si = h_keep[scored[i].second];
                fb[i * 4 + 0] = h_fb[si * 4 + 0];
                fb[i * 4 + 1] = h_fb[si * 4 + 1];
                fb[i * 4 + 2] = h_fb[si * 4 + 2];
                fb[i * 4 + 3] = h_fb[si * 4 + 3];
            }
        }
        if (sb) {
            float *fs = static_cast<float *>(sb);
            memset(fs, 0, cfg.max_detections * sizeof(float));
            for (int i = 0; i < final_n; ++i)
                fs[i] = -scored[i].first;
        }
        if (cb) {
            int *ic = static_cast<int *>(cb);
            memset(ic, -1, cfg.max_detections * sizeof(int));
            for (int i = 0; i < final_n; ++i)
                ic[i] = h_fc[h_keep[scored[i].second]];
        }
    }

    // ---- 3. 异步发送响应（CompletionQueue） ----
    cudaEvent_t event = nullptr;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, stream);

    CompletionTask task;
    task.event = event;
    for (auto &info : infos) {
        if (info.resp) task.responses.push_back(info.resp);
        if (info.sahi_resp) task.sahi_responses.push_back(info.sahi_resp);
        // 转移 det_resps 生命周期到 CompletionTask
        for (auto *r : info.det_resps)
            if (r) task.det_responses.push_back(r);
        info.det_resps.clear();
    }

    is->completion_queue.Push(std::move(task));

    return nullptr;
}

} // extern "C"
