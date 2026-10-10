#include "edge_npu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
