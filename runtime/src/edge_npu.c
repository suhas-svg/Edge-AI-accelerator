#include "edge_npu.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ENPU_MODEL_MAGIC 0x454DU
#define ENPU_MODEL_VERSION 1
#define ENPU_MAC_TILE 8

int enpu_open(enpu_device_t *dev) {
    if (!dev) return -1;
    dev->present = 1;
    return 0;
}

int enpu_load_model(enpu_device_t *dev, const char *path, enpu_model_t *model) {
    if (!dev || !dev->present || !path || !model) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint8_t hdr[12];
    if (fread(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f);
        return -1;
    }
    fclose(f);
    /* Header is little-endian: magic u16, version u16, counts u32/u32. */
    if ((uint16_t)(hdr[0] | (hdr[1] << 8)) != ENPU_MODEL_MAGIC) return -1;
    if ((uint16_t)(hdr[2] | (hdr[3] << 8)) != ENPU_MODEL_VERSION) return -1;
    strncpy(model->path, path, sizeof(model->path) - 1);
    model->path[sizeof(model->path) - 1] = '\0';
    model->version = ENPU_MODEL_VERSION;
    return 0;
}

int enpu_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k) {
    if (!a || !b || !c || m <= 0 || n <= 0 || k <= 0) return -1;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            int32_t acc = 0;
            for (int p = 0; p < k; p++) acc += (int32_t)a[i * k + p] * b[p * n + j];
            c[i * n + j] = acc;
        }
    return 0;
}

int enpu_read_cycles(uint32_t *cycles) {
    if (!cycles) return -1;
    *cycles = 0; /* simulator fills this in; FPGA perf counter later */
    return 0;
}

int enpu_cycles_for_matmul(int m, int n, int k, uint32_t *cycles) {
    if (!cycles || m <= 0 || n <= 0 || k <= 0) return -1;
    if (m % ENPU_MAC_TILE || n % ENPU_MAC_TILE || k % ENPU_MAC_TILE) return -1;
    *cycles = (uint32_t)((int64_t)m * n * k / (ENPU_MAC_TILE * ENPU_MAC_TILE));
    return 0;
}
