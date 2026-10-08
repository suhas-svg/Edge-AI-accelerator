#include "edge_npu.h"
#include <string.h>

int enpu_open(enpu_device_t *dev) {
    if (!dev) return -1;
    dev->present = 1;
    return 0;
}

int enpu_load_model(enpu_device_t *dev, const char *path, enpu_model_t *model) {
    if (!dev || !dev->present || !path || !model) return -1;
    strncpy(model->path, path, sizeof(model->path) - 1);
    model->path[sizeof(model->path) - 1] = '\0';
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
