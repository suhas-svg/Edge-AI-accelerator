#include "edge_npu.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static void write_header(const char *path, uint16_t magic, uint16_t version) {
    FILE *f = fopen(path, "wb");
    assert(f);
    uint8_t hdr[12] = {0};
    hdr[0] = (uint8_t)(magic & 0xff);
    hdr[1] = (uint8_t)(magic >> 8);
    hdr[2] = (uint8_t)(version & 0xff);
    hdr[3] = (uint8_t)(version >> 8);
    assert(fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr));
    fclose(f);
}

int main(void) {
    uint32_t c = 0;
    assert(enpu_cycles_for_matmul(8, 8, 8, &c) == 0 && c == 8);
    assert(enpu_cycles_for_matmul(64, 64, 64, &c) == 0 && c == 4096);
    assert(enpu_cycles_for_matmul(7, 8, 8, &c) != 0);
    assert(enpu_cycles_for_matmul(8, 8, 8, 0) != 0);

    enpu_device_t dev;
    enpu_model_t m;
    assert(enpu_open(&dev) == 0);
    write_header("build/t_good.bin", 0x454D, 1);
    assert(enpu_load_model(&dev, "build/t_good.bin", &m) == 0 && m.version == 1);
    write_header("build/t_badmagic.bin", 0x1234, 1);
    assert(enpu_load_model(&dev, "build/t_badmagic.bin", &m) != 0);
    write_header("build/t_badver.bin", 0x454D, 99);
    assert(enpu_load_model(&dev, "build/t_badver.bin", &m) != 0);
    assert(enpu_load_model(&dev, "build/does-not-exist.bin", &m) != 0);

    assert(ENPU_OK == 0);
    assert(enpu_load_model(&dev, "build/t_badmagic.bin", &m) == ENPU_ERR_VERSION);
    assert(enpu_load_model(&dev, "build/does-not-exist.bin", &m) == ENPU_ERR_IO);
    assert(enpu_open(0) == ENPU_ERR_ARG);

    enpu_buffer_t ba = {0}, bb = {0}, bc = {0};
    assert(enpu_alloc_buffer(&dev, 64, &ba) == ENPU_OK && ba.size == 64 && ba.data);
    assert(enpu_alloc_buffer(&dev, 0, &bb) == ENPU_ERR_ARG);
    assert(enpu_alloc_buffer(0, 64, &bb) == ENPU_ERR_ARG);
    enpu_device_t dead = {0};
    assert(enpu_alloc_buffer(&dead, 64, &bb) == ENPU_ERR_ARG);
    assert(enpu_alloc_buffer(&dev, 64, &bb) == ENPU_OK);
    assert(enpu_alloc_buffer(&dev, 256, &bc) == ENPU_OK);
    for (int i = 0; i < 64; i++) {
        ((int8_t *)ba.data)[i] = (int8_t)(i % 5 - 2);
        ((int8_t *)bb.data)[i] = (int8_t)(i % 3 - 1);
    }
    uint32_t rc = 0;
    assert(enpu_run_matmul(&dev, &ba, &bb, &bc, 8, 8, 8, &rc) == ENPU_OK && rc == 8);
    {
        int32_t expect = 0;
        for (int p = 0; p < 8; p++)
            expect += (int32_t)((int8_t *)ba.data)[p] * ((int8_t *)bb.data)[p * 8];
        assert(((int32_t *)bc.data)[0] == expect);
    }
    assert(enpu_run_matmul(&dev, &ba, &bb, &bc, 7, 8, 8, &rc) == ENPU_ERR_ARG);
    assert(enpu_run_matmul(&dev, &ba, &bb, &bc, 16, 8, 8, &rc) == ENPU_ERR_ARG);
    assert(enpu_run_matmul(0, &ba, &bb, &bc, 8, 8, 8, &rc) == ENPU_ERR_ARG);
    assert(enpu_run_matmul(&dev, 0, &bb, &bc, 8, 8, 8, &rc) == ENPU_ERR_ARG);
    enpu_free_buffer(&ba);
    enpu_free_buffer(&bb);
    enpu_free_buffer(&bc);
    assert(ba.data == 0 && ba.size == 0);
    enpu_free_buffer(&ba);
    enpu_free_buffer(0);

    assert(enpu_reset(0) == ENPU_ERR_ARG);
    assert(enpu_reset(&dead) == ENPU_ERR_ARG);
    assert(enpu_reset(&dev) == ENPU_OK);
    assert(enpu_alloc_buffer(&dev, 64, &bb) == ENPU_ERR_ARG);
    assert(enpu_load_model(&dev, "build/t_good.bin", &m) == ENPU_ERR_ARG);
    assert(enpu_run_matmul(&dev, &ba, &bb, &bc, 8, 8, 8, &rc) == ENPU_ERR_ARG);
    assert(enpu_open(&dev) == ENPU_OK);
    assert(enpu_alloc_buffer(&dev, 64, &bb) == ENPU_OK);
    enpu_free_buffer(&bb);
    assert(enpu_reset(&dev) == ENPU_OK);
    assert(enpu_reset(&dev) == ENPU_ERR_ARG);

    printf("runtime tests ok\n");
    return 0;
}
