#ifndef EDGE_NPU_H
#define EDGE_NPU_H
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int present; } enpu_device_t;
typedef struct { char path[256]; int version; } enpu_model_t;

int enpu_open(enpu_device_t *dev);
int enpu_load_model(enpu_device_t *dev, const char *path, enpu_model_t *model);
int enpu_matmul_i8(const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k);
int enpu_cycles_for_matmul(int m, int n, int k, uint32_t *cycles);
int enpu_read_cycles(uint32_t *cycles);

#ifdef __cplusplus
}
#endif
#endif
