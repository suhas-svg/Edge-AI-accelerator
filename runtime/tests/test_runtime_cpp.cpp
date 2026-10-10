#include "edge_npu_cpp.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <utility>

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

static void expect_code(std::function<void()> fn, int want) {
    try {
        fn();
    } catch (const enpu::Error &e) {
        assert(e.code() == want);
        return;
    }
    assert(!"expected enpu::Error");
}

int main(void) {
    assert(enpu::Device().cycles_for_matmul(8, 8, 8) == 8);
    assert(enpu::Device().cycles_for_matmul(64, 64, 64) == 4096);

    enpu::Device dev;
    write_header("build/cpp_good.bin", 0x454D, 1);
    write_header("build/cpp_badmagic.bin", 0x1234, 1);
    write_header("build/cpp_badver.bin", 0x454D, 99);
    enpu::Model m = dev.load_model("build/cpp_good.bin");
    assert(m.version() == 1);

    expect_code([] { enpu::Device d; d.load_model("build/cpp_badmagic.bin"); },
                ENPU_ERR_VERSION);
    expect_code([] { enpu::Device d; d.load_model("build/cpp_badver.bin"); },
                ENPU_ERR_VERSION);
    expect_code([] { enpu::Device d; d.load_model("build/does-not-exist.bin"); },
                ENPU_ERR_IO);
    expect_code([] { enpu::Device d; d.cycles_for_matmul(7, 8, 8); },
                ENPU_ERR_ARG);
    expect_code([] { enpu::Device d; d.alloc_buffer(0); }, ENPU_ERR_ARG);

    enpu::Buffer ba = dev.alloc_buffer(64);
    enpu::Buffer bb = dev.alloc_buffer(64);
    enpu::Buffer bc = dev.alloc_buffer(256);
    assert(ba.size() == 64 && ba.data() != 0);
    for (int i = 0; i < 64; i++) {
        ((int8_t *)ba.data())[i] = (int8_t)(i % 5 - 2);
        ((int8_t *)bb.data())[i] = (int8_t)(i % 3 - 1);
    }
    assert(dev.run_matmul(ba, bb, bc, 8, 8, 8) == 8);
    {
        int32_t expect = 0;
        for (int p = 0; p < 8; p++)
            expect += (int32_t)((int8_t *)ba.data())[p] * ((int8_t *)bb.data())[p * 8];
        assert(((int32_t *)bc.data())[0] == expect);
    }
    expect_code([&] { dev.run_matmul(ba, bb, bc, 7, 8, 8); }, ENPU_ERR_ARG);

    enpu::Buffer moved = std::move(ba);
    assert(ba.data() == 0 && ba.size() == 0);
    assert(moved.size() == 64);

    dev.reset();
    expect_code([&] { dev.alloc_buffer(64); }, ENPU_ERR_ARG);
    expect_code([&] { dev.load_model("build/cpp_good.bin"); }, ENPU_ERR_ARG);
    expect_code([&] { dev.reset(); }, ENPU_ERR_ARG);
    dev.open();
    enpu::Buffer reopened = dev.alloc_buffer(64);
    assert(reopened.size() == 64);
    enpu::Model m2 = dev.load_model("build/cpp_good.bin");
    assert(m2.version() == 1);

    printf("runtime cpp tests ok\n");
    return 0;
}
