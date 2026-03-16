#include "slang_compiler.h"

#include <slang.h>
#include <slang-com-ptr.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <mutex>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

enum class SlangOutputKind {
    SourceText,
    SpirvBinary,
};

struct SlangTargetConfig {
    SlangCompileTarget format = SLANG_TARGET_NONE;
    const char* profile = nullptr;
    const char* backendLabel = nullptr;
};

void logDiagnostics(const char* prefix, slang::IBlob* diagnostics);

constexpr uint32_t kSpirvHeaderWordCount = 5;
constexpr uint16_t kSpirvOpEntryPoint = 15;
constexpr uint16_t kSpirvOpDecorate = 71;
constexpr uint16_t kSpirvOpMemberDecorate = 72;
constexpr uint16_t kSpirvOpGroupDecorate = 73;
constexpr uint16_t kSpirvOpGroupMemberDecorate = 74;
constexpr uint16_t kSpirvOpDecorateId = 332;
constexpr uint32_t kSpirvExecutionModelFragment = 4;
constexpr uint32_t kSpirvExecutionModelMeshExt = 5365;
constexpr uint32_t kSpirvDecorationLocation = 30;
constexpr uint32_t kSpirvDecorationPerPrimitiveExt = 5271;

std::mutex g_bindingLayoutCacheMutex;
std::unordered_map<uint64_t, SlangShaderBindingLayout> g_bindingLayoutCache;

uint64_t hashBinaryBlob(const void* data, size_t size) {
    constexpr uint64_t kOffsetBasis = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;

    uint64_t hash = kOffsetBasis;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kPrime;
    }
    return hash;
}

uint32_t getDescriptorCount(slang::TypeLayoutReflection* typeLayout,
                            slang::ShaderReflection* /*reflection*/) {
    if (!typeLayout || !typeLayout->isArray()) {
        return 1;
    }

    const size_t count = typeLayout->getElementCount();
    if (count == 0 || count == SLANG_UNBOUNDED_SIZE) {
        return 1;
    }
    return static_cast<uint32_t>(count);
}

bool classifyBindingType(slang::TypeLayoutReflection* typeLayout,
                         SlangShaderBindingType& outType) {
    if (!typeLayout) {
        return false;
    }

    slang::TypeLayoutReflection* leafType = typeLayout->unwrapArray();
    if (!leafType) {
        return false;
    }

    switch (leafType->getKind()) {
    case slang::TypeReflection::Kind::ConstantBuffer:
        outType = SlangShaderBindingType::UniformBuffer;
        return true;
    case slang::TypeReflection::Kind::SamplerState:
        outType = SlangShaderBindingType::Sampler;
        return true;
    case slang::TypeReflection::Kind::Resource:
    case slang::TypeReflection::Kind::ShaderStorageBuffer:
    case slang::TypeReflection::Kind::TextureBuffer:
    case slang::TypeReflection::Kind::DynamicResource: {
        const SlangResourceShape baseShape =
            static_cast<SlangResourceShape>(leafType->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK);
        switch (baseShape) {
        case SLANG_TEXTURE_1D:
        case SLANG_TEXTURE_2D:
        case SLANG_TEXTURE_3D:
        case SLANG_TEXTURE_CUBE:
        case SLANG_TEXTURE_BUFFER:
        case SLANG_TEXTURE_SUBPASS:
            outType = leafType->getResourceAccess() == SLANG_RESOURCE_ACCESS_READ
                ? SlangShaderBindingType::SampledTexture
                : SlangShaderBindingType::StorageTexture;
            return true;
        case SLANG_STRUCTURED_BUFFER:
        case SLANG_BYTE_ADDRESS_BUFFER:
            outType = SlangShaderBindingType::StorageBuffer;
            return true;
        case SLANG_ACCELERATION_STRUCTURE:
            outType = SlangShaderBindingType::AccelerationStructure;
            return true;
        default:
            return false;
        }
    }
    default:
        return false;
    }
}

void cacheBindingLayout(const void* data,
                        size_t size,
                        const SlangShaderBindingLayout& layout) {
    std::lock_guard<std::mutex> lock(g_bindingLayoutCacheMutex);
    g_bindingLayoutCache[hashBinaryBlob(data, size)] = layout;
}

bool buildBindingLayout(slang::IComponentType* linkedProgram,
                        SlangShaderBindingLayout& outLayout) {
    if (!linkedProgram) {
        return false;
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::ProgramLayout* reflection = linkedProgram->getLayout(0, diagnostics.writeRef());
    if (!reflection) {
        logDiagnostics("Slang reflection error", diagnostics);
        return false;
    }

    slang::TypeLayoutReflection* globalParams = reflection->getGlobalParamsTypeLayout();
    if (!globalParams) {
        return true;
    }

    const unsigned fieldCount = globalParams->getFieldCount();
    outLayout.bindings.reserve(fieldCount);
    for (unsigned fieldIndex = 0; fieldIndex < fieldCount; ++fieldIndex) {
        slang::VariableLayoutReflection* field = globalParams->getFieldByIndex(fieldIndex);
        if (!field) {
            continue;
        }

        SlangShaderBindingType bindingType{};
        if (!classifyBindingType(field->getTypeLayout(), bindingType)) {
            continue;
        }

        SlangShaderBindingDesc desc;
        desc.type = bindingType;
        desc.bindingIndex = field->getBindingIndex();
        desc.bindingSpace = static_cast<uint32_t>(field->getBindingSpace());
        desc.descriptorCount = getDescriptorCount(field->getTypeLayout(), reflection);
        if (const char* name = field->getName()) {
            desc.name = name;
        }
        outLayout.bindings.push_back(std::move(desc));
    }

    return true;
}

bool resolveSlangTarget(RhiBackendType backend,
                        SlangOutputKind outputKind,
                        SlangTargetConfig& outConfig) {
    switch (outputKind) {
    case SlangOutputKind::SourceText:
        if (backend == RhiBackendType::Metal) {
            outConfig.format = SLANG_METAL;
            outConfig.profile = "metal";
            outConfig.backendLabel = "Metal";
            return true;
        }
        spdlog::error("Slang source output is not supported for this backend");
        return false;

    case SlangOutputKind::SpirvBinary:
        if (backend == RhiBackendType::Vulkan) {
            outConfig.format = SLANG_SPIRV;
            outConfig.profile = "spirv_1_5";
            outConfig.backendLabel = "Vulkan";
            return true;
        }
        spdlog::error("SPIR-V output is only valid for the Vulkan backend");
        return false;
    }

    return false;
}

bool isSpirvAnnotationOp(uint16_t opcode) {
    switch (opcode) {
    case kSpirvOpDecorate:
    case kSpirvOpMemberDecorate:
    case kSpirvOpGroupDecorate:
    case kSpirvOpGroupMemberDecorate:
    case kSpirvOpDecorateId:
        return true;
    default:
        return false;
    }
}

bool patchSpirvPerPrimitiveFragmentInputs(std::vector<uint32_t>& words,
                                          const char* shaderPath) {
    if (words.size() <= kSpirvHeaderWordCount) {
        return false;
    }

    std::unordered_set<uint32_t> meshInterfaceIds;
    std::unordered_set<uint32_t> fragmentInterfaceIds;
    std::unordered_map<uint32_t, uint32_t> locationsById;
    std::unordered_set<uint32_t> perPrimitiveIds;

    size_t insertWordIndex = 0;

    for (size_t wordIndex = kSpirvHeaderWordCount; wordIndex < words.size();) {
        const uint32_t instruction = words[wordIndex];
        const uint16_t wordCount = static_cast<uint16_t>(instruction >> 16);
        const uint16_t opcode = static_cast<uint16_t>(instruction & 0xFFFFu);
        if (wordCount == 0 || (wordIndex + wordCount) > words.size()) {
            spdlog::warn("Skipping SPIR-V per-primitive patch for {}: malformed module", shaderPath);
            return false;
        }

        if (opcode == kSpirvOpEntryPoint && wordCount >= 4) {
            const uint32_t executionModel = words[wordIndex + 1];
            size_t interfaceWordIndex = wordIndex + 3;
            while (interfaceWordIndex < wordIndex + wordCount) {
                const uint32_t stringWord = words[interfaceWordIndex++];
                if ((stringWord & 0xFF000000u) == 0 ||
                    (stringWord & 0x00FF0000u) == 0 ||
                    (stringWord & 0x0000FF00u) == 0 ||
                    (stringWord & 0x000000FFu) == 0) {
                    break;
                }
            }

            auto& interfaceIds =
                executionModel == kSpirvExecutionModelMeshExt ? meshInterfaceIds : fragmentInterfaceIds;
            if (executionModel == kSpirvExecutionModelMeshExt ||
                executionModel == kSpirvExecutionModelFragment) {
                for (; interfaceWordIndex < wordIndex + wordCount; ++interfaceWordIndex) {
                    interfaceIds.insert(words[interfaceWordIndex]);
                }
            }
        } else if (opcode == kSpirvOpDecorate && wordCount >= 3) {
            const uint32_t targetId = words[wordIndex + 1];
            const uint32_t decoration = words[wordIndex + 2];
            if (decoration == kSpirvDecorationLocation && wordCount >= 4) {
                locationsById[targetId] = words[wordIndex + 3];
            } else if (decoration == kSpirvDecorationPerPrimitiveExt) {
                perPrimitiveIds.insert(targetId);
            }
        }

        if (isSpirvAnnotationOp(opcode)) {
            insertWordIndex = wordIndex + wordCount;
        }

        wordIndex += wordCount;
    }

    std::unordered_set<uint32_t> perPrimitiveLocations;
    for (uint32_t interfaceId : meshInterfaceIds) {
        if (perPrimitiveIds.contains(interfaceId)) {
            auto locationIt = locationsById.find(interfaceId);
            if (locationIt != locationsById.end()) {
                perPrimitiveLocations.insert(locationIt->second);
            }
        }
    }

    if (perPrimitiveLocations.empty() || insertWordIndex == 0) {
        return false;
    }

    std::vector<uint32_t> patchWords;
    for (uint32_t interfaceId : fragmentInterfaceIds) {
        auto locationIt = locationsById.find(interfaceId);
        if (locationIt == locationsById.end()) {
            continue;
        }
        if (!perPrimitiveLocations.contains(locationIt->second) ||
            perPrimitiveIds.contains(interfaceId)) {
            continue;
        }

        patchWords.push_back((3u << 16) | kSpirvOpDecorate);
        patchWords.push_back(interfaceId);
        patchWords.push_back(kSpirvDecorationPerPrimitiveExt);
        perPrimitiveIds.insert(interfaceId);
    }

    if (patchWords.empty()) {
        return false;
    }

    words.insert(words.begin() + static_cast<ptrdiff_t>(insertWordIndex),
                 patchWords.begin(),
                 patchWords.end());
    spdlog::info("Applied SPIR-V per-primitive interface patch for {} ({} fragment inputs)",
                 shaderPath,
                 patchWords.size() / 3u);
    return true;
}

void logDiagnostics(const char* prefix, slang::IBlob* diagnostics) {
    if (!diagnostics) {
        return;
    }

    spdlog::error("{}: {}",
                  prefix,
                  static_cast<const char*>(diagnostics->getBufferPointer()));
}

bool createSession(const SlangTargetConfig& targetConfig,
                   const char* shaderPath,
                   const char* searchPath,
                   Slang::ComPtr<slang::IGlobalSession>& outGlobalSession,
                   Slang::ComPtr<slang::ISession>& outSession) {
    if (SLANG_FAILED(slang::createGlobalSession(outGlobalSession.writeRef()))) {
        spdlog::error("Slang: failed to create global session for {}", shaderPath);
        return false;
    }

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = targetConfig.format;
    targetDesc.profile = outGlobalSession->findProfile(targetConfig.profile);
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    if (SLANG_FAILED(outGlobalSession->createSession(sessionDesc, outSession.writeRef()))) {
        spdlog::error("Slang: failed to create {} session for {}",
                      targetConfig.backendLabel,
                      shaderPath);
        return false;
    }

    return true;
}

slang::IModule* loadModule(slang::ISession* session,
                           const char* shaderPath,
                           const char* searchPath,
                           const char* label,
                           Slang::ComPtr<slang::IBlob>& ioDiagnostics) {
    spdlog::info("Loading {} shader: {} (search path: {})",
                 label,
                 shaderPath,
                 searchPath ? searchPath : "<cwd>");

    slang::IModule* module = session->loadModule(shaderPath, ioDiagnostics.writeRef());
    if (!module) {
        logDiagnostics("Slang load error", ioDiagnostics);
        return nullptr;
    }

    return module;
}

bool buildLinkedProgram(slang::ISession* session,
                        slang::IModule* module,
                        const std::vector<const char*>& entryPoints,
                        const char* shaderPath,
                        Slang::ComPtr<slang::IBlob>& ioDiagnostics,
                        Slang::ComPtr<slang::IComponentType>& outLinkedProgram) {
    std::vector<slang::IComponentType*> components;
    components.reserve(entryPoints.size() + 1);
    components.push_back(module);

    std::vector<Slang::ComPtr<slang::IEntryPoint>> resolvedEntries;
    resolvedEntries.reserve(entryPoints.size());

    for (const char* entryPointName : entryPoints) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        module->findEntryPointByName(entryPointName, entryPoint.writeRef());
        if (!entryPoint) {
            spdlog::error("Slang: entry point '{}' not found in {}", entryPointName, shaderPath);
            return false;
        }

        components.push_back(entryPoint);
        resolvedEntries.push_back(entryPoint);
    }

    Slang::ComPtr<slang::IComponentType> program;
    if (SLANG_FAILED(session->createCompositeComponentType(
            components.data(),
            components.size(),
            program.writeRef(),
            ioDiagnostics.writeRef()))) {
        logDiagnostics("Slang program creation error", ioDiagnostics);
        return false;
    }

    if (SLANG_FAILED(program->link(outLinkedProgram.writeRef(), ioDiagnostics.writeRef()))) {
        logDiagnostics("Slang link error", ioDiagnostics);
        return false;
    }

    return true;
}

std::string compileSlangComponentToSource(RhiBackendType backend,
                                          const char* shaderPath,
                                          const char* searchPath,
                                          const std::vector<const char*>& entryPoints,
                                          const char* label) {
    SlangTargetConfig targetConfig;
    if (!resolveSlangTarget(backend, SlangOutputKind::SourceText, targetConfig)) {
        return {};
    }

    Slang::ComPtr<slang::IGlobalSession> globalSession;
    Slang::ComPtr<slang::ISession> session;
    if (!createSession(targetConfig, shaderPath, searchPath, globalSession, session)) {
        return {};
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = loadModule(session, shaderPath, searchPath, label, diagnostics);
    if (!module) {
        return {};
    }

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    if (!buildLinkedProgram(session, module, entryPoints, shaderPath, diagnostics, linkedProgram)) {
        return {};
    }

    Slang::ComPtr<slang::IBlob> sourceCode;
    if (SLANG_FAILED(linkedProgram->getTargetCode(0, sourceCode.writeRef(), diagnostics.writeRef())) || !sourceCode) {
        logDiagnostics("Slang source compile error", diagnostics);
        return {};
    }

    return std::string(static_cast<const char*>(sourceCode->getBufferPointer()),
                       sourceCode->getBufferSize());
}

std::vector<uint32_t> compileSlangComponentToBinary(RhiBackendType backend,
                                                    const char* shaderPath,
                                                    const char* searchPath,
                                                    const std::vector<const char*>& entryPoints,
                                                    const char* label) {
    SlangTargetConfig targetConfig;
    if (!resolveSlangTarget(backend, SlangOutputKind::SpirvBinary, targetConfig)) {
        return {};
    }

    Slang::ComPtr<slang::IGlobalSession> globalSession;
    Slang::ComPtr<slang::ISession> session;
    if (!createSession(targetConfig, shaderPath, searchPath, globalSession, session)) {
        return {};
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = loadModule(session, shaderPath, searchPath, label, diagnostics);
    if (!module) {
        return {};
    }

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    if (!buildLinkedProgram(session, module, entryPoints, shaderPath, diagnostics, linkedProgram)) {
        return {};
    }

    Slang::ComPtr<slang::IBlob> binaryCode;
    if (SLANG_FAILED(linkedProgram->getTargetCode(0, binaryCode.writeRef(), diagnostics.writeRef())) || !binaryCode) {
        logDiagnostics("Slang binary compile error", diagnostics);
        return {};
    }

    const size_t byteSize = binaryCode->getBufferSize();
    if (byteSize == 0 || (byteSize % sizeof(uint32_t)) != 0) {
        spdlog::error("Slang returned invalid SPIR-V blob size {} for {}", byteSize, shaderPath);
        return {};
    }

    std::vector<uint32_t> words(byteSize / sizeof(uint32_t));
    std::memcpy(words.data(), binaryCode->getBufferPointer(), byteSize);

    patchSpirvPerPrimitiveFragmentInputs(words, shaderPath);

    SlangShaderBindingLayout bindingLayout;
    if (buildBindingLayout(linkedProgram, bindingLayout)) {
        cacheBindingLayout(words.data(), words.size() * sizeof(uint32_t), bindingLayout);
    }
    return words;
}

std::string patchMeshMetalSource(const std::string& source) {
    std::string patched = source;

    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewNormal\w*)\s*;)"),
        "$1 [[user(NORMAL)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewPos\w*)\s*;)"),
        "$1 [[user(TEXCOORD)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((float2\s+\w*uv\w*)\s*;)"),
        "$1 [[user(TEXCOORD_1)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((\[\[flat\]\]\s+uint\s+\w*materialID\w*)\s*;)"),
        "$1 [[user(TEXCOORD_2)]];");

    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+))"),
        "$1 [[texture(0)]]");

    return patched;
}

std::string patchVisibilityMetalSource(const std::string& source) {
    std::string patched = source;

    patched = std::regex_replace(patched,
        std::regex(R"((float2\s+\w*uv\w*)\s*;)"),
        "$1 [[user(TEXCOORD)]];");

    patched = std::regex_replace(patched,
        std::regex(R"((uint\s+\w*visibility\w*)\s*;)"),
        "$1 [[user(TEXCOORD_1)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((uint\s+\w*materialID\w*)\s*;)"),
        "$1 [[user(TEXCOORD_2)]];");

    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+))"),
        "$1 [[texture(0)]]");

    return patched;
}

std::string patchComputeMetalSource(const std::string& source) {
    std::string patched = source;

    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+)(\s*,))"),
        "$1 [[texture(3)]]$2");

    return patched;
}

} // namespace

std::string compileSlangGraphicsSource(RhiBackendType backend,
                                       const char* shaderPath,
                                       const char* searchPath) {
    return compileSlangComponentToSource(backend,
                                         shaderPath,
                                         searchPath,
                                         {"vertexMain", "fragmentMain"},
                                         "graphics");
}

std::string compileSlangMeshSource(RhiBackendType backend,
                                   const char* shaderPath,
                                   const char* searchPath) {
    return compileSlangComponentToSource(backend,
                                         shaderPath,
                                         searchPath,
                                         {"meshMain", "fragmentMain"},
                                         "mesh");
}

std::string compileSlangComputeSource(RhiBackendType backend,
                                      const char* shaderPath,
                                      const char* searchPath,
                                      const char* entryPoint) {
    return compileSlangComponentToSource(backend,
                                         shaderPath,
                                         searchPath,
                                         {entryPoint},
                                         "compute");
}

std::vector<uint32_t> compileSlangGraphicsBinary(RhiBackendType backend,
                                                 const char* shaderPath,
                                                 const char* searchPath) {
    return compileSlangComponentToBinary(backend,
                                         shaderPath,
                                         searchPath,
                                         {"vertexMain", "fragmentMain"},
                                         "graphics");
}

std::vector<uint32_t> compileSlangMeshBinary(RhiBackendType backend,
                                             const char* shaderPath,
                                             const char* searchPath) {
    return compileSlangComponentToBinary(backend,
                                         shaderPath,
                                         searchPath,
                                         {"meshMain", "fragmentMain"},
                                         "mesh");
}

std::vector<uint32_t> compileSlangComputeBinary(RhiBackendType backend,
                                                const char* shaderPath,
                                                const char* searchPath,
                                                const char* entryPoint) {
    return compileSlangComponentToBinary(backend,
                                         shaderPath,
                                         searchPath,
                                         {entryPoint},
                                         "compute");
}

bool findSlangBindingLayoutForBinary(const void* data,
                                     size_t size,
                                     SlangShaderBindingLayout& outLayout) {
    if (!data || size == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_bindingLayoutCacheMutex);
    auto it = g_bindingLayoutCache.find(hashBinaryBlob(data, size));
    if (it == g_bindingLayoutCache.end()) {
        return false;
    }

    outLayout = it->second;
    return true;
}

std::string patchMeshShaderSource(RhiBackendType backend, const std::string& source) {
    return backend == RhiBackendType::Metal ? patchMeshMetalSource(source) : source;
}

std::string patchVisibilityShaderSource(RhiBackendType backend, const std::string& source) {
    return backend == RhiBackendType::Metal ? patchVisibilityMetalSource(source) : source;
}

std::string patchComputeShaderSource(RhiBackendType backend, const std::string& source) {
    return backend == RhiBackendType::Metal ? patchComputeMetalSource(source) : source;
}
