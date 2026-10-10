#ifndef EDGE_NPU_CPP_H
#define EDGE_NPU_CPP_H
#include "edge_npu.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace enpu {

// Error thrown by every wrapper on a nonzero C return; carries the ENPU_* code.
class Error : public std::runtime_error {
public:
    explicit Error(int code);
    int code() const noexcept { return code_; }
private:
    int code_;
};

// RAII owner of an enpu_buffer_t. Move-only; moved-from is empty.
class Buffer {
public:
    Buffer() noexcept;
    ~Buffer();
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept;
    Buffer &operator=(Buffer &&other) noexcept;
    uint8_t *data() const noexcept { return buf_.data; }
    size_t size() const noexcept { return buf_.size; }
    const enpu_buffer_t *get() const noexcept { return &buf_; }
    enpu_buffer_t *get() noexcept { return &buf_; }
private:
    enpu_buffer_t buf_;
    friend class Device;
};

// RAII owner of an opened device. Methods mirror the C API and throw Error.
class Device {
public:
    Device();
    Buffer alloc_buffer(size_t size);
    uint32_t run_matmul(const Buffer &a, const Buffer &b, Buffer &c,
                        int m, int n, int k);
    uint32_t cycles_for_matmul(int m, int n, int k);
    uint32_t read_cycles();
    class Model load_model(const std::string &path);
    const enpu_device_t *get() const noexcept { return &dev_; }
private:
    enpu_device_t dev_;
};

// Loaded model handle. Copyable; holds the validated path and version.
class Model {
public:
    const std::string &path() const noexcept { return path_; }
    int version() const noexcept { return version_; }
private:
    explicit Model(const enpu_model_t &model);
    std::string path_;
    int version_;
    friend class Device;
};

}  // namespace enpu
#endif
