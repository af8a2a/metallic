#pragma once
#include "Evidence.h"
#include "Requirements.h"

namespace metallic::tests::bench {
Json describeDevice(render::Device& device, const Profile& profile);
bool nativeDescriptorPointersEnabled(render::Device& device);
Validation activeValidation(render::Device& device);
} // namespace metallic::tests::bench
