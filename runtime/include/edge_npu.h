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
int enpu_load_model(enpu_device_t *dev, const char *path, enpu_model_t *model);
int enpu_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k);
int enpu_cycles_for_matmul(int m, int n, int k, uint32_t *cycles);
int enpu_alloc_buffer(enpu_device_t *dev, size_t size, enpu_buffer_t *buf);
void enpu_free_buffer(enpu_buffer_t *buf);
int enpu_run_matmul(enpu_device_t *dev, const enpu_buffer_t *a, const enpu_buffer_t *b,
                    enpu_buffer_t *c, int m, int n, int k, uint32_t *cycles);
int enpu_read_cycles(uint32_t *cycles);

#ifdef __cplusplus
}
#endif
#endif
