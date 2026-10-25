#include "Runtime/Render/Streamer/StreamCLASTemplateBounds.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace metallic::render {
namespace {

constexpr double kFloatMin = std::numeric_limits<float>::min();
constexpr double kFloatMax = std::numeric_limits<float>::max();
constexpr double kEpsilon = std::numeric_limits<float>::epsilon();
constexpr float kInfinity = std::numeric_limits<float>::infinity();

struct Interval {
    double lo;
    double hi;
};
using Interval3 = std::array<Interval, 3>;

Interval invalid()
{
    return {-kInfinity, kInfinity};
}

bool finite(Interval value)
{
    return std::isfinite(value.lo) && std::isfinite(value.hi) && value.lo <= value.hi;
}

Interval includeFlushToZero(Interval value)
{
    if (value.lo < kFloatMin && value.hi > -kFloatMin) {
        value.lo = std::min(value.lo, 0.0);
        value.hi = std::max(value.hi, 0.0);
    }
    return value;
}

// Vulkan's full FP32 add/multiply are correctly rounded, division permits
// 2.5 ULP, and input/output denormals may flush. Eight outward float steps
// also cover that allowance across a binade boundary, where adjacent spacing
// halves. They include conversion of the double endpoint and both fused
// and unfused multiply-add: each intermediate includes its exact value too.
// https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#precision-and-operation-of-spir-v-instructions
Interval rounded(double lo, double hi)
{
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi || lo < -kFloatMax || hi > kFloatMax) {
        return invalid();
    }
    float lower = static_cast<float>(lo);
    float upper = static_cast<float>(hi);
    for (unsigned step = 0; step < 8; ++step) {
        lower = std::nextafter(lower, -kInfinity);
        upper = std::nextafter(upper, kInfinity);
    }
    return includeFlushToZero({lower, upper});
}

Interval input(float value)
{
    return includeFlushToZero({value, value});
}

Interval add(Interval a, Interval b)
{
    return rounded(a.lo + b.lo, a.hi + b.hi);
}

Interval subtract(Interval a, Interval b)
{
    return rounded(a.lo - b.hi, a.hi - b.lo);
}

Interval multiply(Interval a, Interval b)
{
    if (!finite(a) || !finite(b)) { return invalid(); }
    const std::array values{a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    return rounded(*std::min_element(values.begin(), values.end()), *std::max_element(values.begin(), values.end()));
}

Interval divide(Interval a, Interval b)
{
    // The SPIR-V 2.5 ULP division guarantee only covers normal divisors up to
    // 2^126. A possible zero, FTZ divisor, or value outside that range cannot
    // establish the finite inverse required by a bounded template.
    constexpr double kLargestDivisor = 0x1p126;
    if (!finite(a) || !finite(b) || (b.lo <= 0 && b.hi >= 0) ||
        std::min(std::abs(b.lo), std::abs(b.hi)) < kFloatMin ||
        std::max(std::abs(b.lo), std::abs(b.hi)) > kLargestDivisor) { return invalid(); }
    const std::array values{a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi};
    return rounded(*std::min_element(values.begin(), values.end()), *std::max_element(values.begin(), values.end()));
}

Interval dot(const Interval3& a, const Interval3& b)
{
    return add(add(multiply(a[0], b[0]), multiply(a[1], b[1])), multiply(a[2], b[2]));
}

Interval3 cross(const Interval3& a, const Interval3& b)
{
    return {subtract(multiply(a[1], b[2]), multiply(a[2], b[1])),
        subtract(multiply(a[2], b[0]), multiply(a[0], b[2])),
        subtract(multiply(a[0], b[1]), multiply(a[1], b[0]))};
}

Interval interpolateSource(Interval worldAxis)
{
    if (!finite(worldAxis)) { return invalid(); }
    // tessUnpackDomain's /4096 patch corners are exact dyadics through depth 3.
    // tessBary divides two packed integers by 65535 and computes the third
    // weight by subtraction. Its deviation from a simplex has L1 error below
    // 24 epsilon when each scalar operation is allowed four ULP. Combining
    // those weights with the three dyadic patch corners (or the edge's
    // step/rate lerp) stays below 256 epsilon L1 error. The final source
    // weighted sum/lerp adds less than 32 epsilon times the coordinate scale.
    // 512 epsilon covers both paths, their fused forms, and roundoff at the
    // zero-component branch. This relies on the current factor <=8/depth <=3
    // domain contract; it must be reviewed if that evaluator changes.
    const double scale = std::max(std::abs(worldAxis.lo), std::abs(worldAxis.hi));
    const double error = 512.0 * kEpsilon * scale + 32.0 * kFloatMin;
    return rounded(worldAxis.lo - error, worldAxis.hi + error);
}

} // namespace

std::optional<StreamCLASTemplateBounds> computeStreamCLASTemplateBounds(
    std::span<const float, 6> allLodBounds,
    std::span<const float, 16> worldMatrix,
    float displacementMagnitude,
    float displacementCenter) noexcept
{
    if (!std::all_of(allLodBounds.begin(), allLodBounds.end(), [](float v) { return std::isfinite(v); }) ||
        !std::all_of(worldMatrix.begin(), worldMatrix.end(), [](float v) { return std::isfinite(v); }) ||
        !std::isfinite(displacementMagnitude) || !std::isfinite(displacementCenter) ||
        displacementCenter < 0.0f || displacementCenter > 1.0f) { return std::nullopt; }
    Interval3 source;
    std::array<Interval3, 3> columns;
    Interval3 translation;
    for (unsigned axis = 0; axis < 3; ++axis) {
        if (allLodBounds[axis] > allLodBounds[axis + 3]) { return std::nullopt; }
        source[axis] = includeFlushToZero({allLodBounds[axis], allLodBounds[axis + 3]});
        translation[axis] = input(worldMatrix[12 + axis]);
        for (unsigned column = 0; column < 3; ++column) { columns[column][axis] = input(worldMatrix[column * 4 + axis]); }
    }

    const std::array cofactors{cross(columns[1], columns[2]), cross(columns[2], columns[0]),
        cross(columns[0], columns[1])};
    const Interval determinant = dot(columns[0], cofactors[0]);
    if (!finite(determinant) || (determinant.lo <= 0 && determinant.hi >= 0)) { return std::nullopt; }

    // tessHeight saturates its sampled value before applying centre/magnitude.
    const Interval height = multiply(subtract({0.0, 1.0}, input(displacementCenter)), input(displacementMagnitude));
    // tessSourceDomain normalizes the interpolated world normal, with a unit
    // geometric/axis fallback. For finite results, 64 epsilon covers the dot,
    // inverse-sqrt (2 ULP) and multiplication error. Its dot threshold 1e-20
    // excludes a subnormal length from the normalization branch.
    constexpr double kNormalComponent = 1.0 + 64.0 * kEpsilon;
    const Interval displacement = multiply(height, {-kNormalComponent, kNormalComponent});
    Interval3 relative;
    for (unsigned axis = 0; axis < 3; ++axis) {
        const Interval3 row{columns[0][axis], columns[1][axis], columns[2][axis]};
        const Interval world = add(dot(row, source), translation[axis]);
        relative[axis] = subtract(add(interpolateSource(world), displacement), translation[axis]);
    }

    StreamCLASTemplateBounds result;
    for (unsigned axis = 0; axis < 3; ++axis) {
        const Interval local = divide(dot(relative, cofactors[axis]), determinant);
        if (!finite(local)) { return std::nullopt; }
        result[axis] = static_cast<float>(local.lo);
        result[axis + 3] = static_cast<float>(local.hi);
    }
    return result;
}

} // namespace metallic::render
