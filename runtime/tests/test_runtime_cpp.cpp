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

    assert(dev.cycles_for_conv2d(8, 8, 8, 8, 3, 3) == 576);
    expect_code([&] { dev.cycles_for_conv2d(7, 8, 8, 8, 3, 3); },
                ENPU_ERR_ARG);

    enpu::Buffer cx = dev.alloc_buffer(800);
    enpu::Buffer cw = dev.alloc_buffer(576);
    enpu::Buffer cy = dev.alloc_buffer(2048);
    for (int i = 0; i < 800; i++)
        ((int8_t *)cx.data())[i] = (int8_t)(i % 16 - 8);
    for (int i = 0; i < 576; i++)
        ((int8_t *)cw.data())[i] = (int8_t)(i % 5 - 2);
    assert(dev.run_conv2d(cx, cw, cy, 8, 10, 10, 8, 3, 3)
           == (uint32_t)(8 * 8 * 8 * 8 * 3 * 3 / 64));
    {
        int32_t expect = 0;
        for (int cc = 0; cc < 8; cc++)
            for (int ki = 0; ki < 3; ki++)
                for (int kj = 0; kj < 3; kj++)
                    expect += (int32_t)((int8_t *)cx.data())[cc * 100 + ki * 10 + kj]
                        * ((int8_t *)cw.data())[cc * 9 + ki * 3 + kj];
        assert(((int32_t *)cy.data())[0] == expect);
    }
    expect_code([&] { dev.run_conv2d(cx, cw, cy, 7, 10, 10, 8, 3, 3); },
                ENPU_ERR_ARG);

    enpu::Buffer ce = dev.alloc_buffer(256);
    enpu::Buffer cb = dev.alloc_buffer(256);
    enpu::Buffer cq = dev.alloc_buffer(64);
    for (int i = 0; i < 64; i++) {
        ((int32_t *)ce.data())[i] = (int32_t)(i % 9 - 4) * 1000;
        ((int32_t *)cb.data())[i] = (int32_t)(i - 32);
    }
    assert(dev.run_relu(ce, 64) == 0);
    assert(((int32_t *)ce.data())[0] == 0 && ((int32_t *)ce.data())[8] == 4000);
    expect_code([&] { dev.run_relu(ce, 63); }, ENPU_ERR_ARG);
    assert(dev.run_bias_add(ce, cb, 64) == 0);
    assert(((int32_t *)ce.data())[8] == 4000 + (8 - 32));
    assert(dev.run_requantize(ce, cq, 64, 0.5f, 3) == 0);
    assert(((int8_t *)cq.data())[8] == 127);
    assert(((int8_t *)cq.data())[0] == -13);
    expect_code([&] { dev.run_requantize(ce, cq, 64, 0.0f, 0); },
                ENPU_ERR_ARG);

    enpu::Buffer px = dev.alloc_buffer(64);
    enpu::Buffer py = dev.alloc_buffer(16);
    for (int i = 0; i < 16; i++)
        ((int32_t *)px.data())[i] = (int32_t)i;
    assert(dev.run_max_pool(px, py, 1, 4, 4, 2, 2) == 0);
    assert(((int32_t *)py.data())[0] == 5 && ((int32_t *)py.data())[3] == 15);
    expect_code([&] { dev.run_max_pool(px, py, 1, 4, 4, 0, 2); },
                ENPU_ERR_ARG);

    printf("runtime cpp tests ok\n");
    return 0;
}
