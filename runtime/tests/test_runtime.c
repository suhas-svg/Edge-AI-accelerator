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

    printf("runtime tests ok\n");
    return 0;
}
