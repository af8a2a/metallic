#pragma once

#include "Runtime/CPUKernels/AABBTransform.h"

#include <array>
#include <cstdint>
#include <vector>

namespace metallic::cpu::test {

struct AABBTestData {
    explicit AABBTestData(uint32_t count)
    {
        for (auto& values : inputs) { values.resize(count); }
        for (auto& values : outputs) { values.resize(count, -12345.0f); }
        uint32_t seed = 0x73b01u;
        const auto randomFloat = [&seed]() {
            seed = seed * 1664525u + 1013904223u;
            return static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);
        };
        for (uint32_t index = 0; index < count; ++index) {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                inputs[axis][index] = randomFloat() * 200.0f - 100.0f;
                inputs[axis + 3][index] = inputs[axis][index] + randomFloat() * 50.0f;
            }
            for (uint32_t element = 0; element < 12; ++element) {
                inputs[element + 6][index] = randomFloat() * 16.0f - 8.0f;
            }
        }
    }

    AABBTransformInput input() const
    {
        AABBTransformInput result;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            result.minimum[axis] = inputs[axis].data();
            result.maximum[axis] = inputs[axis + 3].data();
        }
        for (uint32_t element = 0; element < 12; ++element) {
            result.affine[element] = inputs[element + 6].data();
        }
        return result;
    }

    AABBTransformOutput output()
    {
        AABBTransformOutput result;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            result.minimum[axis] = outputs[axis].data();
            result.maximum[axis] = outputs[axis + 3].data();
        }
        return result;
    }

    std::array<std::vector<float>, 18> inputs;
    std::array<std::vector<float>, 6> outputs;
};

} // namespace metallic::cpu::test
