#include "slang_compiler.h"

#include <slang.h>
#include <slang-com-ptr.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
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

void logDiagnostics(const char* prefix, const char* shaderPath, slang::IBlob* diagnostics);

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
std::mutex g_recentDiagnosticsMutex;
std::vector<SlangDiagnosticRecord> g_recentDiagnostics;
constexpr size_t kMaxRecentDiagnostics = 12;

void recordDiagnostic(const char* stage, const char* shaderPath, std::string message) {
    std::scoped_lock lock(g_recentDiagnosticsMutex);
    SlangDiagnosticRecord record;
    record.stage = stage ? stage : "Slang";
    record.shaderPath = shaderPath ? shaderPath : "";
    record.message = std::move(message);
    g_recentDiagnostics.push_back(std::move(record));
    if (g_recentDiagnostics.size() > kMaxRecentDiagnostics) {
        g_recentDiagnostics.erase(
            g_recentDiagnostics.begin(),
            g_recentDiagnostics.begin() +
                static_cast<std::ptrdiff_t>(g_recentDiagnostics.size() - kMaxRecentDiagnostics));
    }
}

// --- SPIR-V on-disk cache ---

constexpr uint32_t kSpirvMagic = 0x07230203u;
std::string g_shaderCacheDir; // empty = disabled

std::mutex g_compileStatsMutex;
SlangCompileStats g_compileStats;

// --- Include dependency scanning ---

// Resolve a shader file path, trying with .slang extension first.
std::string resolveShaderFilePath(const std::string& basePath) {
    std::string withExt = basePath + ".slang";
    if (std::filesystem::exists(withExt)) {
        return withExt;
    }
    if (std::filesystem::exists(basePath)) {
        return basePath;
    }
    return {};
}

// Scan a shader file for #include "..." directives and recursively collect
// all transitive dependencies. Returns sorted, deduplicated absolute paths.
void collectDependenciesRecursive(const std::string& filePath,
                                  const std::string& searchPath,
                                  std::set<std::string>& visited) {
    std::string absPath;
    try {
        absPath = std::filesystem::canonical(filePath).string();
    } catch (...) {
        return;
    }
    if (visited.count(absPath)) {
        return;
    }
    visited.insert(absPath);

    std::ifstream file(filePath);
    if (!file.is_open()) {
        return;
    }

    // Match #include "relative/path" (not angle-bracket system includes).
    static const std::regex includeRegex(R"RE(^\s*#\s*include\s+"([^"]+)")RE");
    std::string line;
    std::filesystem::path parentDir = std::filesystem::path(filePath).parent_path();

    while (std::getline(file, line)) {
        std::smatch match;
        if (std::regex_search(line, match, includeRegex)) {
            std::string includePath = match[1].str();
            // Resolve relative to the including file's directory first.
            std::filesystem::path resolved = parentDir / includePath;
            if (!std::filesystem::exists(resolved) && !searchPath.empty()) {
                // Fall back to the search path.
                resolved = std::filesystem::path(searchPath) / includePath;
            }
            if (std::filesystem::exists(resolved)) {
                collectDependenciesRecursive(resolved.string(), searchPath, visited);
            }
        }
    }
}

std::vector<std::string> collectShaderDependencies(const char* shaderPath,
                                                    const char* searchPath) {
    std::set<std::string> visited;

    // Try .slang extension first, then bare path.
    std::string rootFile = resolveShaderFilePath(shaderPath);
    if (rootFile.empty()) {
        return {};
    }

    std::string search = searchPath ? searchPath : "";
    collectDependenciesRecursive(rootFile, search, visited);

    // Remove the root file itself — its content is already hashed separately.
    try {
        std::string rootAbs = std::filesystem::canonical(rootFile).string();
        visited.erase(rootAbs);
    } catch (...) {}

    return {visited.begin(), visited.end()};
}

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

// FNV-64 continuation: mix additional bytes into an existing hash.
uint64_t hashMix(uint64_t seed, const void* data, size_t size) {
    constexpr uint64_t kPrime = 1099511628211ull;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        seed ^= bytes[i];
        seed *= kPrime;
    }
    return seed;
}

// Build a stable 64-bit key from backend + shader file content + entry points +
// compile options + transitive include dependencies.
uint64_t computeSpirvCacheKey(RhiBackendType backend,
                               const char* shaderPath,
                               const char* searchPath,
                               const std::vector<const char*>& entryPoints,
                               const SlangCompileOptions* options) {
    constexpr uint64_t kOffsetBasis = 1469598103934665603ull;
    uint64_t h = kOffsetBasis;

    const auto bt = static_cast<uint32_t>(backend);
    h = hashMix(h, &bt, sizeof(bt));

    // Always include the path string so that different shaders with the same
    // entry points get distinct keys even when the file read below fails.
    if (shaderPath) {
        h = hashMix(h, shaderPath, std::strlen(shaderPath));
    }

    // Hash file content so the cache is invalidated when the source changes.
    // Slang's loadModule expects paths without extension — try .slang first.
    {
        std::ifstream file(std::string(shaderPath) + ".slang",
                           std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            file.open(shaderPath, std::ios::binary | std::ios::ate);
        }
        if (file.is_open()) {
            const std::streamsize sz = file.tellg();
            if (sz > 0) {
                std::vector<uint8_t> buf(static_cast<size_t>(sz));
                file.seekg(0, std::ios::beg);
                if (file.read(reinterpret_cast<char*>(buf.data()), sz)) {
                    h = hashMix(h, buf.data(), buf.size());
                }
            }
        }
    }

    for (const char* ep : entryPoints) {
        if (ep) {
            h = hashMix(h, ep, std::strlen(ep) + 1); // include null terminator as separator
        }
    }

    // Hash transitive #include dependencies so edits to shared headers
    // (e.g. bindless_scene.slang, visibility_constants.h) invalidate the cache.
    {
        auto deps = collectShaderDependencies(shaderPath, searchPath);
        for (const auto& depPath : deps) {
            std::ifstream depFile(depPath, std::ios::binary | std::ios::ate);
            if (depFile.is_open()) {
                const std::streamsize sz = depFile.tellg();
                if (sz > 0) {
                    std::vector<uint8_t> buf(static_cast<size_t>(sz));
                    depFile.seekg(0, std::ios::beg);
                    if (depFile.read(reinterpret_cast<char*>(buf.data()), sz)) {
                        h = hashMix(h, buf.data(), buf.size());
                    }
                }
            }
        }
    }

    // Hash compile options so that debug/release or different defines produce
    // separate cache entries.
    if (options) {
        const uint8_t optFlags = (options->optimized ? 1u : 0u) |
                                 (options->generateDebugInfo ? 2u : 0u);
        h = hashMix(h, &optFlags, sizeof(optFlags));
        for (const auto& [key, value] : options->defines) {
            h = hashMix(h, key.data(), key.size());
            const uint8_t sep = '=';
            h = hashMix(h, &sep, 1);
            h = hashMix(h, value.data(), value.size());
            const uint8_t nul = 0;
            h = hashMix(h, &nul, 1);
        }
    }

    return h;
}

// Returns empty vector if not cached or cache file is missing / invalid.
std::vector<uint32_t> tryLoadSpirvCache(uint64_t key) {
    if (g_shaderCacheDir.empty()) {
        return {};
    }
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << key << ".spirv.bin";
    const std::filesystem::path cachePath =
        std::filesystem::path(g_shaderCacheDir) / oss.str();

    std::ifstream in(cachePath, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        return {};
    }
    const std::streamsize byteSize = in.tellg();
    if (byteSize <= 0 || (static_cast<size_t>(byteSize) % sizeof(uint32_t)) != 0) {
        return {};
    }

    std::vector<uint32_t> words(static_cast<size_t>(byteSize) / sizeof(uint32_t));
    in.seekg(0, std::ios::beg);
    if (!in.read(reinterpret_cast<char*>(words.data()), byteSize)) {
        return {};
    }

    // Validate SPIR-V magic number.
    if (words.empty() || words[0] != kSpirvMagic) {
        spdlog::warn("SlangShaderCache: invalid SPIR-V magic in '{}'", cachePath.string());
        return {};
    }
    return words;
}

// Write SPIR-V words atomically: temp file then rename.
void writeSpirvCache(uint64_t key, const std::vector<uint32_t>& words) {
    if (g_shaderCacheDir.empty() || words.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(g_shaderCacheDir, ec);
    if (ec) {
        spdlog::warn("SlangShaderCache: could not create cache dir '{}': {}",
                     g_shaderCacheDir, ec.message());
        return;
    }

    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << key << ".spirv.bin";
    const std::filesystem::path cachePath =
        std::filesystem::path(g_shaderCacheDir) / oss.str();
    const std::string tmpPath = cachePath.string() + ".tmp";

    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        spdlog::warn("SlangShaderCache: could not write '{}'", tmpPath);
        return;
    }
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();

    std::filesystem::rename(tmpPath, cachePath, ec);
    if (ec) {
        std::filesystem::copy_file(tmpPath, cachePath,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmpPath, ec);
    }
}

// --- Binding-layout companion cache (persists Slang reflection alongside SPIR-V) ---

std::filesystem::path bindingLayoutCachePath(uint64_t key) {
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << key << ".layout.bin";
    return std::filesystem::path(g_shaderCacheDir) / oss.str();
}

void writeBindingLayoutCache(uint64_t key, const SlangShaderBindingLayout& layout) {
    if (g_shaderCacheDir.empty()) {
        return;
    }
    const auto path = bindingLayoutCachePath(key);
    const std::string tmpPath = path.string() + ".tmp";

    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return;
    }

    const auto writeU32 = [&](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
    const auto writeU8  = [&](uint8_t v)  { out.write(reinterpret_cast<const char*>(&v), 1); };

    writeU32(static_cast<uint32_t>(layout.bindings.size()));
    for (const auto& b : layout.bindings) {
        writeU8(static_cast<uint8_t>(b.type));
        writeU32(b.bindingIndex);
        writeU32(b.bindingSpace);
        writeU32(b.descriptorCount);
        writeU32(static_cast<uint32_t>(b.name.size()));
        if (!b.name.empty()) {
            out.write(b.name.data(), static_cast<std::streamsize>(b.name.size()));
        }
    }
    out.close();

    std::error_code ec;
    std::filesystem::rename(tmpPath, path, ec);
    if (ec) {
        std::filesystem::copy_file(tmpPath, path,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::filesystem::remove(tmpPath, ec);
    }
}

bool tryLoadBindingLayoutCache(uint64_t key, SlangShaderBindingLayout& outLayout) {
    if (g_shaderCacheDir.empty()) {
        return false;
    }
    const auto path = bindingLayoutCachePath(key);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    const auto readU32 = [&](uint32_t& v) -> bool {
        return !!in.read(reinterpret_cast<char*>(&v), 4);
    };
    const auto readU8 = [&](uint8_t& v) -> bool {
        return !!in.read(reinterpret_cast<char*>(&v), 1);
    };

    uint32_t count = 0;
    if (!readU32(count) || count > 4096) {
        return false;
    }

    outLayout.bindings.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto& b = outLayout.bindings[i];
        uint8_t t = 0;
        uint32_t nameLen = 0;
        if (!readU8(t) || !readU32(b.bindingIndex) || !readU32(b.bindingSpace) ||
            !readU32(b.descriptorCount) || !readU32(nameLen)) {
            return false;
        }
        b.type = static_cast<SlangShaderBindingType>(t);
        if (nameLen > 0) {
            b.name.resize(nameLen);
            if (!in.read(b.name.data(), nameLen)) {
                return false;
            }
        }
    }
    return true;
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
        logDiagnostics("Slang reflection error", nullptr, diagnostics);
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

        bool isPushConstant = field->getCategory() == slang::ParameterCategory::PushConstantBuffer;
        if (!isPushConstant) {
            const unsigned categoryCount = field->getCategoryCount();
            for (unsigned categoryIndex = 0; categoryIndex < categoryCount; ++categoryIndex) {
                if (field->getCategoryByIndex(categoryIndex) == slang::ParameterCategory::PushConstantBuffer) {
                    isPushConstant = true;
                    break;
                }
            }
        }
        if (!isPushConstant) {
            slang::TypeLayoutReflection* typeLayout = field->getTypeLayout();
            if (typeLayout) {
                const SlangInt bindingRangeCount = typeLayout->getBindingRangeCount();
                for (SlangInt rangeIndex = 0; rangeIndex < bindingRangeCount; ++rangeIndex) {
                    if (typeLayout->getBindingRangeType(rangeIndex) == slang::BindingType::PushConstant) {
                        isPushConstant = true;
                        break;
                    }
                }
            }
        }
        if (isPushConstant) {
            SlangShaderBindingDesc desc;
            desc.type = SlangShaderBindingType::PushConstantBuffer;
            desc.bindingIndex = 0;
            desc.bindingSpace = 0;
            desc.descriptorCount = 1;
            if (const char* name = field->getName()) {
                desc.name = name;
            }
            outLayout.bindings.push_back(std::move(desc));
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

void logDiagnostics(const char* prefix, const char* shaderPath, slang::IBlob* diagnostics) {
    if (!diagnostics) {
        return;
    }

    const char* message = static_cast<const char*>(diagnostics->getBufferPointer());
    if (shaderPath && shaderPath[0] != '\0') {
        spdlog::error("{} [{}]: {}", prefix, shaderPath, message);
    } else {
        spdlog::error("{}: {}", prefix, message);
    }
    recordDiagnostic(prefix, shaderPath, message ? message : "");
}

bool createSession(const SlangTargetConfig& targetConfig,
                   const char* shaderPath,
                   const char* searchPath,
                   const SlangCompileOptions* options,
                   Slang::ComPtr<slang::IGlobalSession>& outGlobalSession,
                   Slang::ComPtr<slang::ISession>& outSession) {
    if (SLANG_FAILED(slang::createGlobalSession(outGlobalSession.writeRef()))) {
        spdlog::error("Slang: failed to create global session for {}", shaderPath);
        recordDiagnostic("Slang global session error",
                         shaderPath,
                         "Failed to create Slang global session");
        return false;
    }

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = targetConfig.format;
    targetDesc.profile = outGlobalSession->findProfile(targetConfig.profile);

    // Apply optimization level.
    std::vector<slang::CompilerOptionEntry> compilerOptions;
    if (options) {
        slang::CompilerOptionEntry optLevel;
        optLevel.name = slang::CompilerOptionName::Optimization;
        optLevel.value.kind = slang::CompilerOptionValueKind::Int;
        optLevel.value.intValue0 = options->optimized ? 3 : 0; // -O3 or -O0
        compilerOptions.push_back(optLevel);

        if (options->generateDebugInfo) {
            slang::CompilerOptionEntry debugInfo;
            debugInfo.name = slang::CompilerOptionName::DebugInformation;
            debugInfo.value.kind = slang::CompilerOptionValueKind::Int;
            debugInfo.value.intValue0 = static_cast<int>(SLANG_DEBUG_INFO_LEVEL_STANDARD);
            compilerOptions.push_back(debugInfo);
        }
    }
    if (!compilerOptions.empty()) {
        targetDesc.compilerOptionEntries = compilerOptions.data();
        targetDesc.compilerOptionEntryCount = static_cast<uint32_t>(compilerOptions.size());
    }

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    // Apply preprocessor defines.
    std::vector<slang::PreprocessorMacroDesc> macros;
    if (options) {
        macros.reserve(options->defines.size());
        for (const auto& [key, value] : options->defines) {
            slang::PreprocessorMacroDesc macro;
            macro.name = key.c_str();
            macro.value = value.c_str();
            macros.push_back(macro);
        }
    }
    if (!macros.empty()) {
        sessionDesc.preprocessorMacros = macros.data();
        sessionDesc.preprocessorMacroCount = static_cast<SlangInt>(macros.size());
    }

    if (SLANG_FAILED(outGlobalSession->createSession(sessionDesc, outSession.writeRef()))) {
        spdlog::error("Slang: failed to create {} session for {}",
                      targetConfig.backendLabel,
                      shaderPath);
        recordDiagnostic("Slang session creation error",
                         shaderPath,
                         std::string("Failed to create ") + targetConfig.backendLabel + " session");
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
        logDiagnostics("Slang load error", shaderPath, ioDiagnostics);
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
        logDiagnostics("Slang program creation error", shaderPath, ioDiagnostics);
        return false;
    }

    if (SLANG_FAILED(program->link(outLinkedProgram.writeRef(), ioDiagnostics.writeRef()))) {
        logDiagnostics("Slang link error", shaderPath, ioDiagnostics);
        return false;
    }

    return true;
}

std::string compileSlangComponentToSource(RhiBackendType backend,
                                          const char* shaderPath,
                                          const char* searchPath,
                                          const std::vector<const char*>& entryPoints,
                                          const char* label,
                                          const SlangCompileOptions* options) {
    SlangTargetConfig targetConfig;
    if (!resolveSlangTarget(backend, SlangOutputKind::SourceText, targetConfig)) {
        return {};
    }

    Slang::ComPtr<slang::IGlobalSession> globalSession;
    Slang::ComPtr<slang::ISession> session;
    if (!createSession(targetConfig, shaderPath, searchPath, options, globalSession, session)) {
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
        logDiagnostics("Slang source compile error", shaderPath, diagnostics);
        return {};
    }

    return std::string(static_cast<const char*>(sourceCode->getBufferPointer()),
                       sourceCode->getBufferSize());
}

std::vector<uint32_t> compileSlangComponentToBinary(RhiBackendType backend,
                                                    const char* shaderPath,
                                                    const char* searchPath,
                                                    const std::vector<const char*>& entryPoints,
                                                    const char* label,
                                                    const SlangCompileOptions* options) {
    // Check the on-disk SPIR-V cache before invoking Slang.
    const uint64_t cacheKey = computeSpirvCacheKey(backend, shaderPath, searchPath, entryPoints, options);
    {
        std::vector<uint32_t> cached = tryLoadSpirvCache(cacheKey);
        if (!cached.empty()) {
            spdlog::debug("SlangShaderCache: cache hit for '{}' ({} words)", shaderPath, cached.size());
            // Restore the binding layout into the in-memory cache so that
            // findSlangBindingLayoutForBinary() succeeds for callers.
            SlangShaderBindingLayout layout;
            if (tryLoadBindingLayoutCache(cacheKey, layout)) {
                cacheBindingLayout(cached.data(), cached.size() * sizeof(uint32_t), layout);
            }
            {
                std::lock_guard<std::mutex> lock(g_compileStatsMutex);
                g_compileStats.cacheHits++;
            }
            return cached;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_compileStatsMutex);
        g_compileStats.cacheMisses++;
    }

    const auto compileStart = std::chrono::steady_clock::now();

    SlangTargetConfig targetConfig;
    if (!resolveSlangTarget(backend, SlangOutputKind::SpirvBinary, targetConfig)) {
        return {};
    }

    Slang::ComPtr<slang::IGlobalSession> globalSession;
    Slang::ComPtr<slang::ISession> session;
    if (!createSession(targetConfig, shaderPath, searchPath, options, globalSession, session)) {
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
        logDiagnostics("Slang binary compile error", shaderPath, diagnostics);
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
        writeBindingLayoutCache(cacheKey, bindingLayout);
    }

    // Persist to disk so subsequent runs skip Slang compilation.
    writeSpirvCache(cacheKey, words);

    const auto compileEnd = std::chrono::steady_clock::now();
    const float compileMs = std::chrono::duration<float, std::milli>(compileEnd - compileStart).count();
    {
        std::lock_guard<std::mutex> lock(g_compileStatsMutex);
        g_compileStats.compileCount++;
        g_compileStats.totalCompileTimeMs += compileMs;
    }
    spdlog::debug("SlangShaderCache: compiled and cached '{}' ({} words, {:.1f} ms)",
                  shaderPath, words.size(), compileMs);

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

void setSlangShaderCacheDir(const std::string& dir) {
    g_shaderCacheDir = dir;
}

std::string compileSlangGraphicsSource(RhiBackendType backend,
                                       const char* shaderPath,
                                       const char* searchPath,
                                       const SlangCompileOptions* options) {
    return compileSlangComponentToSource(backend,
                                         shaderPath,
                                         searchPath,
                                         {"vertexMain", "fragmentMain"},
                                         "graphics",
                                         options);
}

std::string compileSlangMeshSource(RhiBackendType backend,
                                   const char* shaderPath,
                                   const char* searchPath,
                                   const SlangCompileOptions* options) {
    return compileSlangComponentToSource(backend,
                                         shaderPath,
                                         searchPath,
                                         {"meshMain", "fragmentMain"},
                                         "mesh",
                                         options);
}

std::string compileSlangComputeSource(RhiBackendType backend,
                                      const char* shaderPath,
                                      const char* searchPath,
                                      const char* entryPoint,
                                      const SlangCompileOptions* options) {
    return compileSlangComponentToSource(backend,
                                         shaderPath,
                                         searchPath,
                                         {entryPoint},
                                         "compute",
                                         options);
}

std::vector<uint32_t> compileSlangGraphicsBinary(RhiBackendType backend,
                                                 const char* shaderPath,
                                                 const char* searchPath,
                                                 const SlangCompileOptions* options) {
    return compileSlangComponentToBinary(backend,
                                         shaderPath,
                                         searchPath,
                                         {"vertexMain", "fragmentMain"},
                                         "graphics",
                                         options);
}

std::vector<uint32_t> compileSlangMeshBinary(RhiBackendType backend,
                                             const char* shaderPath,
                                             const char* searchPath,
                                             const SlangCompileOptions* options) {
    return compileSlangComponentToBinary(backend,
                                         shaderPath,
                                         searchPath,
                                         {"meshMain", "fragmentMain"},
                                         "mesh",
                                         options);
}

std::vector<uint32_t> compileSlangComputeBinary(RhiBackendType backend,
                                                const char* shaderPath,
                                                const char* searchPath,
                                                const char* entryPoint,
                                                const SlangCompileOptions* options) {
    return compileSlangComponentToBinary(backend,
                                         shaderPath,
                                         searchPath,
                                         {entryPoint},
                                         "compute",
                                         options);
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

std::vector<SlangDiagnosticRecord> getRecentSlangDiagnostics() {
    std::scoped_lock lock(g_recentDiagnosticsMutex);
    return g_recentDiagnostics;
}

void clearRecentSlangDiagnostics() {
    std::scoped_lock lock(g_recentDiagnosticsMutex);
    g_recentDiagnostics.clear();
}

SlangCompileStats getSlangCompileStats() {
    std::lock_guard<std::mutex> lock(g_compileStatsMutex);
    return g_compileStats;
}

void resetSlangCompileStats() {
    std::lock_guard<std::mutex> lock(g_compileStatsMutex);
    g_compileStats = {};
}
