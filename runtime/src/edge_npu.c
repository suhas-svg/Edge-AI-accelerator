#include "edge_npu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ENPU_MODEL_MAGIC 0x454DU
#define ENPU_MODEL_VERSION 1
#define ENPU_MAC_TILE 8

static int dev_live(const enpu_device_t *dev) {
    return dev && dev->present;
}

int enpu_open(enpu_device_t *dev) {
    if (!dev) return ENPU_ERR_ARG;
    dev->present = 1;
    return ENPU_OK;
}

int enpu_reset(enpu_device_t *dev) {
    if (!dev_live(dev)) return ENPU_ERR_ARG;
    dev->present = 0;
    return ENPU_OK;
}

int enpu_load_model(enpu_device_t *dev, const char *path, enpu_model_t *model) {
    if (!dev_live(dev) || !path || !model) return ENPU_ERR_ARG;
    FILE *f = fopen(path, "rb");
    if (!f) return ENPU_ERR_IO;
    uint8_t hdr[12];
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f);
        return ENPU_ERR_IO;
    }
    fclose(f);
    /* Header is little-endian: magic u16, version u16, counts u32/u32. */
    if ((uint16_t)(hdr[0] | (hdr[1] << 8)) != ENPU_MODEL_MAGIC) return ENPU_ERR_VERSION;
    if ((uint16_t)(hdr[2] | (hdr[3] << 8)) != ENPU_MODEL_VERSION) return ENPU_ERR_VERSION;
    strncpy(model->path, path, sizeof(model->path) - 1);
    model->path[sizeof(model->path) - 1] = '\0';
    model->version = ENPU_MODEL_VERSION;
    return ENPU_OK;
}

int enpu_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) return ENPU_ERR_ARG;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int p = 0; p < k; p++) acc += (int32_t)a[i * k + p] * b[p * n + j];
            c[i * n + j] = acc;
        }
    return 0;
}

int enpu_read_cycles(uint32_t *cycles) {
    if (!cycles) return ENPU_ERR_ARG;
    *cycles = 0; /* simulator fills this in; FPGA perf counter later */
    return ENPU_OK;
}

int enpu_cycles_for_matmul(int m, int n, int k, uint32_t *cycles) {
    if (!cycles || m <= 0 || n <= 0 || k <= 0) return ENPU_ERR_ARG;
    if (m % ENPU_MAC_TILE || n % ENPU_MAC_TILE || k % ENPU_MAC_TILE) return ENPU_ERR_ARG;
    *cycles = (uint32_t)((int64_t)m * n * k / (ENPU_MAC_TILE * ENPU_MAC_TILE));
    return ENPU_OK;
}

int enpu_alloc_buffer(enpu_device_t *dev, size_t size, enpu_buffer_t *buf) {
    if (!dev_live(dev) || !buf || size == 0) return ENPU_ERR_ARG;
    buf->data = (uint8_t *)calloc(1, size);
    if (!buf->data) {
        buf->size = 0;
        return ENPU_ERR_NOMEM;
    }
    buf->size = size;
    return ENPU_OK;
}

void enpu_free_buffer(enpu_buffer_t *buf) {
    if (!buf) return;
    free(buf->data);
    buf->data = 0;
    buf->size = 0;
}

int enpu_run_matmul(enpu_device_t *dev, const enpu_buffer_t *a, const enpu_buffer_t *b,
                    enpu_buffer_t *c, int m, int n, int k, uint32_t *cycles) {
    if (!dev_live(dev) || !a || !b || !c || !cycles) return ENPU_ERR_ARG;
    if (m <= 0 || n <= 0 || k <= 0) return ENPU_ERR_ARG;
    if (m % ENPU_MAC_TILE || n % ENPU_MAC_TILE || k % ENPU_MAC_TILE) return ENPU_ERR_ARG;
    if (!a->data || a->size < (size_t)m * k) return ENPU_ERR_ARG;
    if (!b->data || b->size < (size_t)k * n) return ENPU_ERR_ARG;
    if (!c->data || c->size < (size_t)m * n * sizeof(int32_t)) return ENPU_ERR_ARG;
    int rc = enpu_matmul_i8((const int8_t *)a->data, (const int8_t *)b->data,
                            (int32_t *)c->data, m, n, k);
    if (rc != ENPU_OK) return rc;
    return enpu_cycles_for_matmul(m, n, k, cycles);
}

int enpu_relu_i32(int32_t *buf, size_t n) {
    if (!buf || n == 0) return ENPU_ERR_ARG;
    for (size_t i = 0; i < n; i++)
        if (buf[i] < 0) buf[i] = 0;
    return ENPU_OK;
}

int enpu_bias_add_i32(int32_t *buf, const int32_t *bias, size_t n) {
    if (!buf || !bias || n == 0) return ENPU_ERR_ARG;
    for (size_t i = 0; i < n; i++) buf[i] += bias[i];
    return ENPU_OK;
}

int enpu_requantize_i32(const int32_t *acc, int8_t *out, size_t n, float scale,
                        int zero_point) {
    if (!acc || !out || n == 0) return ENPU_ERR_ARG;
    if (!(scale > 0)) return ENPU_ERR_ARG;
    if (zero_point < -128 || zero_point > 127) return ENPU_ERR_ARG;
    for (size_t i = 0; i < n; i++) {
        int32_t q = (int32_t)rintf((float)acc[i] * scale) + zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        out[i] = (int8_t)q;
    }
    return ENPU_OK;
}

int enpu_max_pool_i32(const int32_t *x, int32_t *y, int c, int h, int w,
                      int size, int stride) {
    if (!x || !y || c <= 0 || h <= 0 || w <= 0) return ENPU_ERR_ARG;
    if (size <= 0 || stride <= 0) return ENPU_ERR_ARG;
    if (size > h || size > w) return ENPU_ERR_ARG;
    int oh = (h - size) / stride + 1, ow = (w - size) / stride + 1;
    for (int cc = 0; cc < c; cc++)
        for (int i = 0; i < oh; i++)
            for (int j = 0; j < ow; j++) {
                int32_t best = x[(cc * h + i * stride) * w + j * stride];
                for (int ki = 0; ki < size; ki++)
                    for (int kj = 0; kj < size; kj++) {
                        int32_t v = x[(cc * h + i * stride + ki) * w + j * stride + kj];
                        if (v > best) best = v;
                    }
                y[(cc * oh + i) * ow + j] = best;
            }
    return ENPU_OK;
}

int enpu_conv2d_i8(const int8_t *x, const int8_t *w, int32_t *y,
                   int c, int h, int wd, int k, int kh, int kw) {
    if (!x || !w || !y) return ENPU_ERR_ARG;
    if (c <= 0 || h <= 0 || wd <= 0 || k <= 0 || kh <= 0 || kw <= 0)
        return ENPU_ERR_ARG;
    if (kh != kw) return ENPU_ERR_ARG;
    int oh = h - kh + 1, ow = wd - kw + 1;
    if (oh <= 0 || ow <= 0) return ENPU_ERR_ARG;
    for (int f = 0; f < k; f++)
        for (int i = 0; i < oh; i++)
            for (int j = 0; j < ow; j++) {
                int32_t acc = 0;
                for (int cc = 0; cc < c; cc++)
                    for (int ki = 0; ki < kh; ki++)
                        for (int kj = 0; kj < kw; kj++)
                            acc += (int32_t)x[(cc * h + i + ki) * wd + j + kj]
                                * w[((f * c) + cc) * kh * kw + ki * kw + kj];
                y[(f * oh + i) * ow + j] = acc;
            }
    return ENPU_OK;
}

int enpu_cycles_for_conv2d(int k, int oh, int ow, int c, int kh, int kw,
                           uint32_t *cycles) {
    if (!cycles || k <= 0 || oh <= 0 || ow <= 0 || c <= 0 || kh <= 0 || kw <= 0)
        return ENPU_ERR_ARG;
    if (k % ENPU_MAC_TILE || c % ENPU_MAC_TILE || oh % ENPU_MAC_TILE || ow % ENPU_MAC_TILE)
        return ENPU_ERR_ARG;
    *cycles = (uint32_t)((int64_t)k * oh * ow * c * kh * kw
                         / (ENPU_MAC_TILE * ENPU_MAC_TILE));
    return ENPU_OK;
}

int enpu_run_relu(enpu_device_t *dev, enpu_buffer_t *buf, size_t n, uint32_t *cycles) {
    if (!dev_live(dev) || !buf || !cycles) return ENPU_ERR_ARG;
    if (n == 0 || n % (ENPU_MAC_TILE * ENPU_MAC_TILE)) return ENPU_ERR_ARG;
    if (!buf->data || buf->size < n * sizeof(int32_t)) return ENPU_ERR_ARG;
    int rc = enpu_relu_i32((int32_t *)buf->data, n);
    if (rc != ENPU_OK) return rc;
    *cycles = 0;
    return ENPU_OK;
}

int enpu_run_bias_add(enpu_device_t *dev, enpu_buffer_t *buf, const enpu_buffer_t *bias,
                      size_t n, uint32_t *cycles) {
    if (!dev_live(dev) || !buf || !bias || !cycles) return ENPU_ERR_ARG;
    if (n == 0 || n % (ENPU_MAC_TILE * ENPU_MAC_TILE)) return ENPU_ERR_ARG;
    if (!buf->data || buf->size < n * sizeof(int32_t)) return ENPU_ERR_ARG;
    if (!bias->data || bias->size < n * sizeof(int32_t)) return ENPU_ERR_ARG;
    int rc = enpu_bias_add_i32((int32_t *)buf->data, (const int32_t *)bias->data, n);
    if (rc != ENPU_OK) return rc;
    *cycles = 0;
    return ENPU_OK;
}

int enpu_run_requantize(enpu_device_t *dev, const enpu_buffer_t *acc, enpu_buffer_t *out,
                        size_t n, float scale, int zero_point, uint32_t *cycles) {
    if (!dev_live(dev) || !acc || !out || !cycles) return ENPU_ERR_ARG;
    if (n == 0 || n % (ENPU_MAC_TILE * ENPU_MAC_TILE)) return ENPU_ERR_ARG;
    if (!acc->data || acc->size < n * sizeof(int32_t)) return ENPU_ERR_ARG;
    if (!out->data || out->size < n) return ENPU_ERR_ARG;
    int rc = enpu_requantize_i32((const int32_t *)acc->data, (int8_t *)out->data,
                                 n, scale, zero_point);
    if (rc != ENPU_OK) return rc;
    *cycles = 0;
    return ENPU_OK;
}

int enpu_run_max_pool(enpu_device_t *dev, const enpu_buffer_t *x, enpu_buffer_t *y,
                      int c, int h, int w, int size, int stride, uint32_t *cycles) {
    if (!dev_live(dev) || !x || !y || !cycles) return ENPU_ERR_ARG;
    if (c <= 0 || h <= 0 || w <= 0 || size <= 0 || stride <= 0) return ENPU_ERR_ARG;
    int oh = (h - size) / stride + 1, ow = (w - size) / stride + 1;
    if (oh <= 0 || ow <= 0) return ENPU_ERR_ARG;
    if (!x->data || x->size < (size_t)c * h * w * sizeof(int32_t)) return ENPU_ERR_ARG;
    if (!y->data || y->size < (size_t)c * oh * ow * sizeof(int32_t)) return ENPU_ERR_ARG;
    int rc = enpu_max_pool_i32((const int32_t *)x->data, (int32_t *)y->data,
                               c, h, w, size, stride);
    if (rc != ENPU_OK) return rc;
    *cycles = 0;
    return ENPU_OK;
}

int enpu_run_conv2d(enpu_device_t *dev, const enpu_buffer_t *x, const enpu_buffer_t *w, enpu_buffer_t *y,
                    int c, int h, int wd, int k, int kh, int kw, uint32_t *cycles) {
    if (!dev_live(dev) || !x || !w || !y || !cycles) return ENPU_ERR_ARG;
    if (c <= 0 || h <= 0 || wd <= 0 || k <= 0 || kh <= 0 || kw <= 0) return ENPU_ERR_ARG;
    if (k % ENPU_MAC_TILE || c % ENPU_MAC_TILE) return ENPU_ERR_ARG;
    int oh = h - kh + 1, ow = wd - kw + 1;
    if (oh <= 0 || ow <= 0) return ENPU_ERR_ARG;
    if (oh % ENPU_MAC_TILE || ow % ENPU_MAC_TILE) return ENPU_ERR_ARG;
    if (!x->data || x->size < (size_t)c * h * wd) return ENPU_ERR_ARG;
    if (!w->data || w->size < (size_t)k * c * kh * kw) return ENPU_ERR_ARG;
    if (!y->data || y->size < (size_t)k * oh * ow * sizeof(int32_t)) return ENPU_ERR_ARG;
    int rc = enpu_conv2d_i8((const int8_t *)x->data, (const int8_t *)w->data,
                            (int32_t *)y->data, c, h, wd, k, kh, kw);
    if (rc != ENPU_OK) return rc;
    return enpu_cycles_for_conv2d(k, oh, ow, c, kh, kw, cycles);
}
