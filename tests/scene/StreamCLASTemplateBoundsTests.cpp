#include "Runtime/Render/Streamer/StreamCLASTemplateBounds.h"
#include "Runtime/Render/TessellationPatterns.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace metallic::render {
namespace {

using Vector = std::array<float, 3>;
using Matrix = std::array<float, 16>;
constexpr Matrix kIdentity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

// Scalar evaluation is intentionally independent of the interval helper. The
// volatile intermediates preserve unfused FP32 operations in optimized builds;
// the other path explicitly contracts with fma. Both can model input/output FTZ.
struct FloatMath {
    bool fused;
    bool ftz;

    float flush(float value) const
    {
        return ftz && std::abs(value) < std::numeric_limits<float>::min() ? std::copysign(0.0f, value) : value;
    }
    float add(float a, float b) const
    {
        volatile float result = flush(a) + flush(b);
        return flush(result);
    }
    float subtract(float a, float b) const { return add(a, -b); }
    float multiply(float a, float b) const
    {
        volatile float result = flush(a) * flush(b);
        return flush(result);
    }
    float divide(float a, float b) const
    {
        volatile float result = flush(a) / flush(b);
        return flush(result);
    }
    float madd(float a, float b, float c) const
    {
        return fused ? flush(std::fma(flush(a), flush(b), flush(c))) : add(multiply(a, b), c);
    }
    float lerp(float a, float b, float t) const
    {
        return fused ? madd(subtract(b, a), t, a) : add(multiply(a, subtract(1, t)), multiply(b, t));
    }
    float dot(const Vector& a, const Vector& b) const
    {
        return madd(a[2], b[2], madd(a[1], b[1], multiply(a[0], b[0])));
    }
    Vector cross(const Vector& a, const Vector& b) const
    {
        return {madd(a[1], b[2], -multiply(a[2], b[1])),
            madd(a[2], b[0], -multiply(a[0], b[2])), madd(a[0], b[1], -multiply(a[1], b[0]))};
    }
};

Vector worldPoint(const Matrix& matrix, Vector point, FloatMath math)
{
    Vector result;
    for (unsigned axis = 0; axis < 3; ++axis) {
        result[axis] = math.madd(matrix[8 + axis], point[2],
            math.madd(matrix[4 + axis], point[1], math.madd(matrix[axis], point[0], matrix[12 + axis])));
    }
    return result;
}

Vector localPoint(const Matrix& matrix, Vector point, FloatMath math)
{
    const Vector c0{matrix[0], matrix[1], matrix[2]};
    const Vector c1{matrix[4], matrix[5], matrix[6]};
    const Vector c2{matrix[8], matrix[9], matrix[10]};
    const std::array cofactors{math.cross(c1, c2), math.cross(c2, c0), math.cross(c0, c1)};
    const float determinant = math.dot(c0, cofactors[0]);
    for (unsigned axis = 0; axis < 3; ++axis) { point[axis] = math.subtract(point[axis], matrix[12 + axis]); }
    return {math.divide(math.dot(point, cofactors[0]), determinant),
        math.divide(math.dot(point, cofactors[1]), determinant), math.divide(math.dot(point, cofactors[2]), determinant)};
}

Vector sourcePoint(const std::array<Vector, 3>& source, Vector bary, FloatMath math)
{
    const unsigned zero = bary[2] == 0 ? 2 : bary[0] == 0 ? 0 : bary[1] == 0 ? 1 : 3;
    Vector result;
    if (zero < 3) {
        unsigned a = (zero + 1) % 3, b = (zero + 2) % 3;
        if (source[a] > source[b]) { std::swap(a, b); }
        for (unsigned axis = 0; axis < 3; ++axis) { result[axis] = math.lerp(source[a][axis], source[b][axis], bary[b]); }
    } else {
        for (unsigned axis = 0; axis < 3; ++axis) {
            result[axis] = math.madd(source[2][axis], bary[2],
                math.madd(source[1][axis], bary[1], math.multiply(source[0][axis], bary[0])));
        }
    }
    return result;
}

Vector patchBary(const std::array<uint32_t, 3>& packed, const std::array<uint32_t, 3>& rates,
    const std::array<Vector, 3>& patch, const std::array<Vector, 3>& source, FloatMath math)
{
    const float y = math.divide(float(packed[1]), 65535), z = math.divide(float(packed[2]), 65535);
    const Vector local{math.subtract(math.subtract(1, y), z), y, z};
    const unsigned zero = local[2] < 1e-7f ? 2 : local[0] < 1e-7f ? 0 : local[1] < 1e-7f ? 1 : 3;
    Vector result;
    if (zero < 3) {
        unsigned a = (zero + 1) % 3, b = (zero + 2) % 3;
        const unsigned rate = rates[a];
        unsigned step = unsigned(std::round(math.multiply(local[b], float(rate))));
        if (sourcePoint(source, patch[a], math) > sourcePoint(source, patch[b], math)) {
            std::swap(a, b); step = rate - step;
        }
        const float t = math.divide(float(step), float(rate));
        for (unsigned axis = 0; axis < 3; ++axis) { result[axis] = math.lerp(patch[a][axis], patch[b][axis], t); }
    } else {
        for (unsigned axis = 0; axis < 3; ++axis) {
            result[axis] = math.madd(patch[2][axis], local[2],
                math.madd(patch[1][axis], local[1], math.multiply(patch[0][axis], local[0])));
        }
    }
    return result;
}

bool contains(const StreamCLASTemplateBounds& bounds, Vector position)
{
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(position[axis]) || position[axis] < bounds[axis] || position[axis] > bounds[axis + 3]) { return false; }
    }
    return true;
}

TEST(StreamCLASTemplateBounds, IdentityEnvelopeRemainsTightAndHandlesSignedDisplacement)
{
    const StreamCLASTemplateBounds source{-1, -2, -.5f, 1, 2, .5f};
    for (float magnitude : {-.4f, .4f}) {
        const auto bounds = computeStreamCLASTemplateBounds(source, kIdentity, magnitude, .25f);
        ASSERT_TRUE(bounds);
        for (unsigned axis = 0; axis < 3; ++axis) {
            EXPECT_LE((*bounds)[axis], source[axis] - .3f);
            EXPECT_GE((*bounds)[axis + 3], source[axis + 3] + .3f);
            EXPECT_GT((*bounds)[axis], source[axis] - .301f);
            EXPECT_LT((*bounds)[axis + 3], source[axis + 3] + .301f);
        }
    }
}

TEST(StreamCLASTemplateBounds, ContainsAllPatternsUnderAffineTransformsAndFusedOrFlushedEvaluation)
{
    const StreamCLASTemplateBounds sourceBounds{-1, -2, -.5f, 1, 2, .5f};
    const std::array<Vector, 3> source{Vector{-1, -2, -.5f}, Vector{1, -1, .5f}, Vector{.2f, 2, .4f}};
    const std::array patches{
        std::array{Vector{1, 0, 0}, Vector{0, 1, 0}, Vector{0, 0, 1}},
        std::array{Vector{.5f, .5f, 0}, Vector{0, .5f, .5f}, Vector{.5f, 0, .5f}},
        std::array{Vector{1, 0, 0}, Vector{.875f, .125f, 0}, Vector{.875f, 0, .125f}}};
    const std::array matrices{kIdentity,
        Matrix{-2, 0, 0, 0, 0, 3, 0, 0, 0, 0, .25f, 0, 3, -2, 1, 1},
        Matrix{.5f, .4f, -.2f, 0, -.3f, 1.2f, .1f, 0, .2f, -.15f, .8f, 0, 11, -17, 5, 1},
        Matrix{.01f, 0, 0, 0, 0, .03f, 0, 0, 0, 0, .02f, 0, 1000000, -1000000, 500000, 1}};
    for (size_t matrixIndex = 0; matrixIndex < matrices.size(); ++matrixIndex) {
        SCOPED_TRACE(matrixIndex);
        const auto& matrix = matrices[matrixIndex];
        const auto bounds = computeStreamCLASTemplateBounds(sourceBounds, matrix, -.4f, .25f);
        ASSERT_TRUE(bounds);
        for (bool fused : {false, true}) {
            for (bool ftz : {false, true}) {
                const FloatMath math{fused, ftz};
                std::array<Vector, 3> world;
                for (unsigned corner = 0; corner < 3; ++corner) { world[corner] = worldPoint(matrix, source[corner], math); }
                for (uint32_t a = 1; a <= 8; ++a) {
                    for (uint32_t b = 1; b <= 8; ++b) {
                        for (uint32_t c = 1; c <= 8; ++c) {
                            const auto pattern = makeTessellationPattern({a, b, c});
                            for (const auto& patch : patches) {
                                for (size_t vertex = 0; vertex < pattern.vertices.size(); ++vertex) {
                                    const auto bary = patchBary(pattern.vertices[vertex], {a, b, c}, patch, world, math);
                                    auto position = sourcePoint(world, bary, math);
                                    Vector normal{vertex % 2 ? -.3f : .6f, .7f, vertex % 3 ? .4f : -.8f};
                                    const float norm = std::sqrt(math.dot(normal, normal));
                                    const float height = math.multiply(math.subtract(float(vertex % 17) / 16.f, .25f), -.4f);
                                    for (unsigned axis = 0; axis < 3; ++axis) {
                                        normal[axis] = math.divide(normal[axis], norm);
                                        position[axis] = math.madd(normal[axis], height, position[axis]);
                                    }
                                    if (!contains(*bounds, localPoint(matrix, position, math))) {
                                        FAIL() << "Finite shader-style point escaped bounds: pattern=" << a << ',' << b << ',' << c
                                            << " vertex=" << vertex << " fused=" << fused << " ftz=" << ftz;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(StreamCLASTemplateBounds, CoversWorldTranslationCancellationBeyondTheIdealLocalBox)
{
    const StreamCLASTemplateBounds source{1, 1, 1, 2, 2, 2};
    auto matrix = kIdentity;
    matrix[12] = matrix[13] = matrix[14] = 100000000.f;
    const auto bounds = computeStreamCLASTemplateBounds(source, matrix, 0, .5f);
    ASSERT_TRUE(bounds);
    const FloatMath math{false, false};
    const auto actual = localPoint(matrix, worldPoint(matrix, {1, 1, 1}, math), math);
    EXPECT_EQ(actual, (Vector{0, 0, 0}));
    EXPECT_TRUE(contains(*bounds, actual));
    EXPECT_LT((*bounds)[0], source[0]);
}

TEST(StreamCLASTemplateBounds, ContainsSubnormalInputsWithAndWithoutFlushToZero)
{
    const float tiny = std::numeric_limits<float>::denorm_min() * 32;
    const StreamCLASTemplateBounds source{-tiny, -tiny, -tiny, tiny, tiny, tiny};
    auto matrix = kIdentity;
    matrix[12] = tiny;
    const auto bounds = computeStreamCLASTemplateBounds(source, matrix, tiny, .5f);
    ASSERT_TRUE(bounds);
    for (bool ftz : {false, true}) {
        const FloatMath math{true, ftz};
        EXPECT_TRUE(contains(*bounds, localPoint(matrix, worldPoint(matrix, {tiny, -tiny, tiny}, math), math)));
    }
}

TEST(StreamCLASTemplateBounds, ReturnsUnrestrictedForUnprovableInverseOrInvalidInput)
{
    const StreamCLASTemplateBounds source{-1, -1, -1, 1, 1, 1};
    auto matrix = kIdentity;
    matrix[0] = 0;
    EXPECT_FALSE(computeStreamCLASTemplateBounds(source, matrix, .4f, .5f));
    matrix = kIdentity;
    matrix[0] = matrix[1] = matrix[4] = 1;
    matrix[5] = std::nextafter(1.f, 2.f);
    EXPECT_FALSE(computeStreamCLASTemplateBounds(source, matrix, .4f, .5f));
    matrix = kIdentity;
    matrix[0] = matrix[5] = matrix[10] = 1e-20f;
    EXPECT_FALSE(computeStreamCLASTemplateBounds(source, matrix, .4f, .5f));
    matrix = kIdentity;
    matrix[0] = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(computeStreamCLASTemplateBounds(source, matrix, .4f, .5f));
    EXPECT_FALSE(computeStreamCLASTemplateBounds(source, kIdentity, std::numeric_limits<float>::quiet_NaN(), .5f));
    EXPECT_FALSE(computeStreamCLASTemplateBounds(source, kIdentity, .4f, -1));
    auto reversed = source;
    reversed[0] = 2;
    EXPECT_FALSE(computeStreamCLASTemplateBounds(reversed, kIdentity, .4f, .5f));
    auto overflow = source;
    overflow[3] = std::numeric_limits<float>::max();
    EXPECT_FALSE(computeStreamCLASTemplateBounds(overflow, kIdentity, .4f, .5f));
}

} // namespace
} // namespace metallic::render
