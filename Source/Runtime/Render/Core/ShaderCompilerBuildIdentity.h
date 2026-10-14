#pragma once

#include <string_view>

namespace metallic::render {

struct ShaderCompilerBuildIdentity
{
    std::string_view sha256;
    bool allowCrossHostReuse = false;
};

// The provider is generated after the shared compiler archive is built, and is
// linked separately so its own identity bytes cannot enter that archive's hash.
// Unsupported toolchain/link contracts retain host-specific cache identities.
ShaderCompilerBuildIdentity compiledShaderCompilerBuildIdentity() noexcept;

} // namespace metallic::render
