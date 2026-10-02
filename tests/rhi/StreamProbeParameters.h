#pragma once
#include "../../Source/Runtime/Render/Core/StreamSceneParameters.h"
#ifdef __cplusplus
namespace metallic::tests {
using StreamProbeScene = render::StreamSceneParameters;
using StreamProbeOutput = render::ShaderDataSpan;
#else
using Metallic;
typealias StreamProbeScene = StreamSceneParameters;
typealias StreamProbeOutput = DataSpan<uint>;
#endif
struct StreamProbeParameters {
    StreamProbeScene stream;
    StreamProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamProbeABI = 0x5354525052420001ull;
static_assert(sizeof(StreamProbeParameters) == 80);
} // namespace metallic::tests
#endif
