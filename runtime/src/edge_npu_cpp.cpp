#include "edge_npu_cpp.h"
#include <string>

namespace enpu {

namespace {
void throw_on_error(int rc) {
    if (rc != ENPU_OK)
        throw Error(rc);
}
}  // namespace

Error::Error(int code)
    : std::runtime_error("EdgeNPU error " + std::to_string(code)), code_(code) {}

Buffer::Buffer() noexcept {
    buf_.data = 0;
    buf_.size = 0;
}

Buffer::~Buffer() {
    enpu_free_buffer(&buf_);
}

Buffer::Buffer(Buffer &&other) noexcept : buf_(other.buf_) {
    other.buf_.data = 0;
    other.buf_.size = 0;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
    if (this != &other) {
        enpu_free_buffer(&buf_);
        buf_ = other.buf_;
        other.buf_.data = 0;
        other.buf_.size = 0;
    }
    return *this;
}

Device::Device() {
    throw_on_error(enpu_open(&dev_));
}

void Device::open() {
    throw_on_error(enpu_open(&dev_));
}

void Device::reset() {
    throw_on_error(enpu_reset(&dev_));
}

Buffer Device::alloc_buffer(size_t size) {
    Buffer buf;
    throw_on_error(enpu_alloc_buffer(&dev_, size, buf.get()));
    return buf;
}

uint32_t Device::run_matmul(const Buffer &a, const Buffer &b, Buffer &c,
                            int m, int n, int k) {
    uint32_t cycles = 0;
    throw_on_error(enpu_run_matmul(&dev_, a.get(), b.get(), c.get(), m, n, k, &cycles));
    return cycles;
}

uint32_t Device::cycles_for_matmul(int m, int n, int k) {
    uint32_t cycles = 0;
    throw_on_error(enpu_cycles_for_matmul(m, n, k, &cycles));
    return cycles;
}

uint32_t Device::read_cycles() {
    uint32_t cycles = 0;
    throw_on_error(enpu_read_cycles(&cycles));
    return cycles;
}

Model Device::load_model(const std::string &path) {
    enpu_model_t raw;
    throw_on_error(enpu_load_model(&dev_, path.c_str(), &raw));
    return Model(raw);
}

Model::Model(const enpu_model_t &model)
    : path_(model.path), version_(model.version) {}

}  // namespace enpu
