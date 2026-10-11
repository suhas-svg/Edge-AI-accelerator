#ifndef EDGE_NPU_H
#define EDGE_NPU_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int present; } enpu_device_t;
typedef struct { char path[256]; int version; } enpu_model_t;
typedef struct { uint8_t *data; size_t size; } enpu_buffer_t;

enum {
    ENPU_OK = 0,
    ENPU_ERR_ARG = -1,
    ENPU_ERR_NOMEM = -2,
    ENPU_ERR_IO = -3,
    ENPU_ERR_VERSION = -4
};

int enpu_open(enpu_device_t *dev);
int enpu_reset(enpu_device_t *dev);
int enpu_load_model(enpu_device_t *dev, const char *path, enpu_model_t *model);
int enpu_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k);
int enpu_relu_i32(int32_t *buf, size_t n);
int enpu_bias_add_i32(int32_t *buf, const int32_t *bias, size_t n);
int enpu_requantize_i32(const int32_t *acc, int8_t *out, size_t n, float scale,
                        int zero_point);
int enpu_max_pool_i32(const int32_t *x, int32_t *y, int c, int h, int w,
                      int size, int stride);
int enpu_conv2d_i8(const int8_t *x, const int8_t *w, int32_t *y,
                   int c, int h, int wd, int k, int kh, int kw);
int enpu_cycles_for_matmul(int m, int n, int k, uint32_t *cycles);
int enpu_cycles_for_conv2d(int k, int oh, int ow, int c, int kh, int kw,
                           uint32_t *cycles);
int enpu_alloc_buffer(enpu_device_t *dev, size_t size, enpu_buffer_t *buf);
void enpu_free_buffer(enpu_buffer_t *buf);
int enpu_run_matmul(enpu_device_t *dev, const enpu_buffer_t *a, const enpu_buffer_t *b,
                    enpu_buffer_t *c, int m, int n, int k, uint32_t *cycles);
int enpu_run_relu(enpu_device_t *dev, enpu_buffer_t *buf, size_t n, uint32_t *cycles);
int enpu_run_bias_add(enpu_device_t *dev, enpu_buffer_t *buf, const enpu_buffer_t *bias,
                      size_t n, uint32_t *cycles);
int enpu_run_requantize(enpu_device_t *dev, const enpu_buffer_t *acc, enpu_buffer_t *out,
                        size_t n, float scale, int zero_point, uint32_t *cycles);
int enpu_run_max_pool(enpu_device_t *dev, const enpu_buffer_t *x, enpu_buffer_t *y,
                      int c, int h, int w, int size, int stride, uint32_t *cycles);
int enpu_run_conv2d(enpu_device_t *dev, const enpu_buffer_t *x, const enpu_buffer_t *w, enpu_buffer_t *y,
                    int c, int h, int wd, int k, int kh, int kw, uint32_t *cycles);
int enpu_read_cycles(uint32_t *cycles);

#ifdef __cplusplus
}
#endif
#endif
