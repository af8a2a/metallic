#include "AABBTestData.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

#if defined(_WIN64)
#include <windows.h>
#endif

namespace {

using namespace metallic::cpu;

constexpr std::array kBackends{CPUKernelBackend::Scalar,
    CPUKernelBackend::AutoVectorized, CPUKernelBackend::ISPC, CPUKernelBackend::Automatic};

void expectDoubleCornerBounds(const test::AABBTestData& data, uint32_t begin, uint32_t end)
{
    for (uint32_t index = begin; index < end; ++index) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            double referenceMin = std::numeric_limits<double>::infinity();
            double referenceMax = -std::numeric_limits<double>::infinity();
            double scale = std::abs(static_cast<double>(data.inputs[6 + axis * 4 + 3][index]));
            for (uint32_t sourceAxis = 0; sourceAxis < 3; ++sourceAxis) {
                scale += std::abs(static_cast<double>(data.inputs[6 + axis * 4 + sourceAxis][index])) *
                    std::max(std::abs(static_cast<double>(data.inputs[sourceAxis][index])),
                        std::abs(static_cast<double>(data.inputs[sourceAxis + 3][index])));
            }
            for (uint32_t corner = 0; corner < 8; ++corner) {
                double position = data.inputs[6 + axis * 4 + 3][index];
                for (uint32_t sourceAxis = 0; sourceAxis < 3; ++sourceAxis) {
                    const uint32_t source = sourceAxis + ((corner & (1 << sourceAxis)) ? 3 : 0);
                    position += static_cast<double>(data.inputs[6 + axis * 4 + sourceAxis][index]) *
                        data.inputs[source][index];
                }
                referenceMin = std::min(referenceMin, position);
                referenceMax = std::max(referenceMax, position);
            }
            const double tolerance = 1e-6 + 8 * std::numeric_limits<float>::epsilon() * scale;
            EXPECT_NEAR(data.outputs[axis][index], referenceMin, tolerance) << "index=" << index;
            EXPECT_NEAR(data.outputs[axis + 3][index], referenceMax, tolerance) << "index=" << index;
        }
    }
}

TEST(CPUKernels, EmptyRangesNeedNoStorage)
{
    for (CPUKernelBackend backend : kBackends) {
        transformAABBs({}, {}, 0, 0, backend);
        transformAABBs({}, {}, UINT32_MAX, UINT32_MAX, backend);
    }
}

TEST(CPUKernels, RandomAffineBoundsMatchDoubleCornerOracle)
{
    // These ranges exercise scalar tails for both 4- and 8-wide gangs, including
    // a nonzero/misaligned start and a zero-length interior range.
    for (uint32_t count : {1u, 3u, 4u, 7u, 8u, 9u, 31u, 1031u}) {
        for (CPUKernelBackend backend : kBackends) {
            SCOPED_TRACE(cpuKernelBackendName(backend));
            test::AABBTestData data(count + 4);
            transformAABBs(data.input(), data.output(), 2, count + 2, backend);
            expectDoubleCornerBounds(data, 2, count + 2);
            for (const auto& values : data.outputs) {
                EXPECT_EQ(values[0], -12345.0f);
                EXPECT_EQ(values[1], -12345.0f);
                EXPECT_EQ(values[count + 2], -12345.0f);
                EXPECT_EQ(values[count + 3], -12345.0f);
            }
        }
    }
}

TEST(CPUKernels, HandlesDegenerateReflectedShearedAndExtremeFiniteInputs)
{
    for (CPUKernelBackend backend : kBackends) {
        test::AABBTestData data(8);
        for (uint32_t index = 0; index < 8; ++index) {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                const float magnitude = index == 6 ? 1e15f : (index == 7 ? 1e-15f : 4.0f);
                data.inputs[axis][index] = -magnitude;
                data.inputs[axis + 3][index] = index == 0 ? -magnitude : magnitude;
                for (uint32_t column = 0; column < 4; ++column) {
                    float coefficient = column == axis ? 1.0f : 0.0f;
                    if (index == 1) { coefficient = -coefficient; }
                    if (index == 2 && column != 3) { coefficient = float(column + axis) - 1.5f; }
                    if (index == 3) { coefficient = 0.0f; }
                    if (index == 4 && column == 3) { coefficient = -100.0f; }
                    if (index == 5 && column == axis) { coefficient = -0.0f; }
                    data.inputs[6 + axis * 4 + column][index] = coefficient;
                }
            }
        }
        transformAABBs(data.input(), data.output(), 0, 8, backend);
        expectDoubleCornerBounds(data, 0, 8);
    }
}

TEST(CPUKernels, ReportsTheBackendThatActuallyRan)
{
    test::AABBTestData data(1);
    EXPECT_EQ(transformAABBs(data.input(), data.output(), 0, 1, CPUKernelBackend::Scalar),
        CPUKernelBackend::Scalar);
    EXPECT_EQ(transformAABBs(data.input(), data.output(), 0, 1, CPUKernelBackend::AutoVectorized),
        CPUKernelBackend::AutoVectorized);
    const CPUKernelBackend expected = cpuKernelBackendAvailable(CPUKernelBackend::ISPC)
        ? CPUKernelBackend::ISPC : CPUKernelBackend::AutoVectorized;
    EXPECT_EQ(transformAABBs(data.input(), data.output(), 0, 1, CPUKernelBackend::ISPC), expected);
    EXPECT_EQ(transformAABBs(data.input(), data.output(), 0, 1), CPUKernelBackend::AutoVectorized);
}

#if defined(_WIN64)
TEST(CPUKernels, LargeArrayOffsetsAndUint32TailRetainAllAddressBits)
{
    // Reserve sparse arrays, committing just their last page. This exercises
    // real >4 GiB byte offsets and a UINT32_MAX tail without resident huge data.
    struct SparseArray {
        ~SparseArray()
        {
            if (base) { VirtualFree(base, 0, MEM_RELEASE); }
        }
        float* base = nullptr;
    };
    std::array<SparseArray, 7> storage{};
    SYSTEM_INFO systemInfo;
    GetSystemInfo(&systemInfo);
    constexpr uint32_t begin = UINT32_MAX - 15, end = UINT32_MAX;
    constexpr size_t kArrayBytes = (uint64_t(UINT32_MAX) + 1) * sizeof(float);
    const size_t commitOffset = (uint64_t(begin) * sizeof(float) / systemInfo.dwPageSize) * systemInfo.dwPageSize;
    for (SparseArray& array : storage) {
        array.base = static_cast<float*>(VirtualAlloc(nullptr, kArrayBytes, MEM_RESERVE, PAGE_NOACCESS));
        ASSERT_NE(array.base, nullptr);
        void* page = reinterpret_cast<char*>(array.base) + commitOffset;
        ASSERT_NE(VirtualAlloc(page, systemInfo.dwPageSize, MEM_COMMIT, PAGE_READWRITE), nullptr);
    }
    for (uint32_t index = begin; index < end; ++index) { storage[0].base[index] = 1.0f; }
    AABBTransformInput input;
    input.minimum.fill(storage[0].base);
    input.maximum.fill(storage[0].base);
    input.affine.fill(storage[0].base);
    AABBTransformOutput output;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        output.minimum[axis] = storage[axis + 1].base;
        output.maximum[axis] = storage[axis + 4].base;
    }
    for (CPUKernelBackend backend : kBackends) {
        SCOPED_TRACE(cpuKernelBackendName(backend));
        for (uint32_t axis = 0; axis < 3; ++axis) {
            output.minimum[axis][begin - 1] = output.minimum[axis][end] = -12345.0f;
            output.maximum[axis][begin - 1] = output.maximum[axis][end] = -12345.0f;
            for (uint32_t index = begin; index < end; ++index) {
                output.minimum[axis][index] = output.maximum[axis][index] = -12345.0f;
            }
        }
        transformAABBs(input, output, begin, end, backend);
        for (uint32_t index = begin; index < end; ++index) {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                EXPECT_EQ(output.minimum[axis][index], 4.0f);
                EXPECT_EQ(output.maximum[axis][index], 4.0f);
            }
        }
        for (uint32_t axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(output.minimum[axis][begin - 1], -12345.0f);
            EXPECT_EQ(output.minimum[axis][end], -12345.0f);
            EXPECT_EQ(output.maximum[axis][begin - 1], -12345.0f);
            EXPECT_EQ(output.maximum[axis][end], -12345.0f);
        }
    }
}
#endif

} // namespace
