#include "Runtime/Render/GAPI/PipelineStateHash.h"

#include <cstring>
#include <type_traits>

namespace metallic::render::detail {
namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
// Increment when an implicit RHI pipeline state changes without a desc change.
constexpr uint32_t kPipelineStateHashVersion = 3;
constexpr uint32_t kGraphicsPipelineTag = 0x4750534fu;
constexpr uint32_t kComputePipelineTag = 0x4350534fu;

uint64_t hashBytes(uint64_t hash, const void* data, size_t byteSize)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < byteSize; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
uint64_t hashValue(uint64_t hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return hashBytes(hash, &value, sizeof(T));
}

uint64_t hashString(uint64_t hash, const char* value)
{
    const char* text = value != nullptr ? value : "main";
    const uint64_t length = std::strlen(text);
    hash = hashValue(hash, length);
    return hashBytes(hash, text, static_cast<size_t>(length));
}

uint8_t hashBool(bool value)
{
    return value ? 1u : 0u;
}

} // namespace

uint64_t shaderContentHash(const ShaderModuleDesc& desc)
{
    uint64_t hash = kFnvOffset;
    hash = hashValue(hash, desc.byteSize);
    return desc.code != nullptr && desc.byteSize > 0
        ? hashBytes(hash, desc.code, static_cast<size_t>(desc.byteSize))
        : hash;
}

uint64_t graphicsPipelineStateHash(const GraphicsPipelineDesc& desc)
{
    uint64_t hash = kFnvOffset;
    hash = hashValue(hash, kPipelineStateHashVersion);
    hash = hashValue(hash, kGraphicsPipelineTag);
    const bool usesMeshShader = desc.meshShader != nullptr;
    hash = hashValue(hash, hashBool(usesMeshShader));
    if (usesMeshShader) {
        hash = hashValue(hash, desc.meshShader->contentHash());
        hash = hashString(hash, desc.meshEntryPoint);
    } else {
        hash = hashValue(hash, desc.vertexShader != nullptr ? desc.vertexShader->contentHash() : 0ull);
        hash = hashString(hash, desc.vertexEntryPoint);
    }
    hash = hashValue(hash, desc.fragmentShader != nullptr ? desc.fragmentShader->contentHash() : 0ull);
    hash = hashString(hash, desc.fragmentEntryPoint);
    hash = hashValue(hash, static_cast<uint32_t>(desc.colorFormat));
    hash = hashValue(hash, static_cast<uint32_t>(desc.depthStencilFormat));
    hash = hashValue(hash, static_cast<uint32_t>(desc.topology));
    hash = hashValue(hash, static_cast<uint32_t>(desc.rasterization.cullMode));
    hash = hashValue(hash, static_cast<uint32_t>(desc.rasterization.frontFace));
    hash = hashValue(hash, hashBool(desc.depthStencil.depthTestEnable));
    hash = hashValue(hash, hashBool(desc.depthStencil.depthWriteEnable));
    hash = hashValue(hash, static_cast<uint32_t>(desc.depthStencil.depthCompareOp));
    hash = hashValue(hash, hashBool(desc.usesBindlessHeap));
    return hash;
}

uint64_t computePipelineStateHash(const ComputePipelineDesc& desc)
{
    uint64_t hash = kFnvOffset;
    hash = hashValue(hash, kPipelineStateHashVersion);
    hash = hashValue(hash, kComputePipelineTag);
    hash = hashValue(hash, desc.computeShader != nullptr ? desc.computeShader->contentHash() : 0ull);
    hash = hashString(hash, desc.computeEntryPoint);
    hash = hashValue(hash, hashBool(desc.usesBindlessHeap));
    hash = hashValue(hash, desc.bindlessUserPushDataSize);
    hash = hashValue(hash, desc.bindingMappingCount);
    if (desc.bindingMappings != nullptr) {
        for (uint32_t index = 0; index < desc.bindingMappingCount; ++index) {
            const ShaderBindingMappingDesc& mapping = desc.bindingMappings[index];
            hash = hashValue(hash, mapping.descriptorSet);
            hash = hashValue(hash, mapping.firstBinding);
            hash = hashValue(hash, mapping.bindingCount);
            hash = hashValue(hash, static_cast<uint32_t>(mapping.type));
            hash = hashValue(hash, static_cast<uint32_t>(mapping.source));
            hash = hashValue(hash, mapping.pushDataOffset);
        }
    }
    return hash;
}

} // namespace metallic::render::detail
