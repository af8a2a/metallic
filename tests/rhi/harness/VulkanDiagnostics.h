#pragma once
#include "Evidence.h"
#include "Requirements.h"

namespace metallic::tests::bench {
Json describeDevice(render::Device& device, const Profile& profile);
bool validationActive(render::Device& device);
} // namespace metallic::tests::bench
