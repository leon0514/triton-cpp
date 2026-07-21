#include <cuda_runtime.h>

class AutoDevice
{
public:
    explicit AutoDevice(int device_id)
    {
        cudaGetDevice(&prev_device_);
        if (prev_device_ != device_id)
        {
            cudaSetDevice(device_id);
            switched_ = true;
        }
    }

    ~AutoDevice()
    {
        if (switched_)
        {
            cudaSetDevice(prev_device_);
        }
    }

    // 禁止拷贝和赋值
    AutoDevice(const AutoDevice &) = delete;
    AutoDevice &operator=(const AutoDevice &) = delete;

private:
    int prev_device_ = 0;
    bool switched_ = false;
};

/**
 * @brief 将输入 buffer 拷贝到当前 GPU workspace，支持 CPU、同 GPU、跨 GPU 三种来源。
 *
 * @param dst              目标设备指针（位于 dst_device_id 上）
 * @param src              源指针
 * @param bytes            拷贝字节数
 * @param src_on_device    源是否在 GPU 上
 * @param src_device_id    源 GPU 号（仅当 src_on_device 为 true 时有效）
 * @param dst_device_id    目标 GPU 号（即当前 CUDA 实例的 device_id）
 * @param stream           CUDA 流
 *
 * @return cudaError_t CUDA 错误码
 */
inline cudaError_t
CopyBufferToDevice(
    void *dst,
    const void *src,
    size_t bytes,
    bool src_on_device,
    int src_device_id,
    int dst_device_id,
    cudaStream_t stream)
{
    if (bytes == 0)
    {
        return cudaSuccess;
    }

    if (!src_on_device)
    {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    }

    if (src_device_id == dst_device_id)
    {
        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    }

    return cudaMemcpyPeerAsync(dst, dst_device_id, src, src_device_id, bytes, stream);
}
