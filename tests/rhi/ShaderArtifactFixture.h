#pragma once

#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/GAPI/Hash.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Tools/SPIRVStatistics.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <json.hpp>

namespace metallic::tests {

// Test-only immutable bytecode fixture. It never populates Slang's disk cache.
class ShaderArtifactFixture final : public render::ShaderArtifactOverride
{
public:
    using Json = nlohmann::json;

    explicit ShaderArtifactFixture(const std::filesystem::path& path)
        : manifestPath_(std::filesystem::absolute(path).lexically_normal())
    {
        const auto bytes = readFile(manifestPath_);
        manifestSha256_ = debug::debugSha256(bytes);
        manifest_ = Json::parse(bytes.begin(), bytes.end());
        require(manifest_.at("version").get<uint32_t>() == 1 && !manifest_.at("artifact_id").get<std::string>().empty(), "Invalid artifact manifest identity");
        const auto& request = manifest_.at("reference_request");
        const auto& target = manifest_.at("expected_target");
        require(target.at("module").get<std::string>() == request.at("module").get<std::string>() &&
            target.at("entry").get<std::string>() == request.at("entry").get<std::string>(), "Artifact target differs from reference request");
        require(request.at("module").get<std::string>() == "Features/PathTracing/OpenPBRRayQueryPathTraceGuides" &&
            request.at("entry").get<std::string>() == "openPbrRayQueryPathTraceGuidesMain" && request.at("profile").get<std::string>() == "spirv_1_6",
            "Artifact fixture supports only the fixed OpenPBR Guides request");
        const auto& settings = manifest_.at("reference_settings");
        require(settings.at("optimization").get<std::string>() == "O1" && settings.at("descriptor").get<std::string>() == "mapped" &&
            settings.at("debug").get<std::string>() == "disabled" && settings.at("working_space").get<std::string>() == "acescg" &&
            settings.at("postprocess").get<std::string>() == "none",
            "Unsupported artifact reference settings");
        const auto& artifact = manifest_.at("artifact");
        artifactPath_ = resolve(artifact.at("file").get<std::string>());
        const auto raw = readFile(artifactPath_);
        require(raw.size() >= 20 && raw.size() % sizeof(uint32_t) == 0 && raw.size() <= 64 * 1024 * 1024 &&
            raw.size() == artifact.at("bytes").get<size_t>(), "Artifact SPIR-V size mismatch");
        sha256_ = debug::debugSha256(raw);
        fnv1a64_ = hashHex(raw);
        require(sha256_ == artifact.at("sha256").get<std::string>() && fnv1a64_ == artifact.at("fnv1a64").get<std::string>(), "Artifact SPIR-V SHA256/FNV mismatch");
        words_.resize(raw.size() / sizeof(uint32_t));
        std::memcpy(words_.data(), raw.data(), raw.size());
        const auto stats = tools::inspectSPIRV(words_);
        require(stats.functions > 0 && stats.basicBlocks > 0 && stats.instructions > 0 &&
            words_[1] <= 0x00010600u, "Artifact SPIR-V structure/version unsupported");
        statistics_ = {{"instructions", stats.instructions}, {"functions", stats.functions}, {"basic_blocks", stats.basicBlocks},
            {"conditional_branches", stats.conditionalBranches}, {"function_calls", stats.functionCalls}, {"id_bound", stats.idBound}};
        const auto entry = artifact.at("entry_main").get<std::string>();
        require(entry == "main" && !artifact.at("source_tag").get<std::string>().empty(), "Missing artifact entry/source provenance");
        validatePostprocessProvenance(artifact);
        if (postprocess() == "none") {
            require(stats.functions == 1 && stats.functionCalls == 0, "SSA artifact must retain one function and zero calls");
        }
        std::set<uint32_t> capabilities;
        size_t entries = 0;
        for (size_t offset = 5; offset < words_.size();) {
            const uint32_t count = words_[offset] >> 16, opcode = words_[offset] & 0xffff;
            if (opcode == 17) { require(count == 2, "Malformed artifact capability"); capabilities.insert(words_[offset + 1]); }
            if (opcode == 15) {
                require(count >= 4 && words_[offset + 1] == 5, "Artifact entry must be compute");
                const char* name = reinterpret_cast<const char*>(words_.data() + offset + 3);
                const size_t budget = (count - 3) * sizeof(uint32_t);
                const char* end = static_cast<const char*>(std::memchr(name, '\0', budget));
                require(end && std::string_view(name, size_t(end - name)) == entry, "Artifact native entry mismatch");
                ++entries;
            }
            offset += count;
        }
        require(entries == 1 && capabilities.contains(1), "Artifact requires one Shader compute entry");
        for (const auto& capability : request.at("capabilities")) {
            const auto name = capability.get<std::string>();
            if (name == "spvRayQueryKHR") { require(capabilities.contains(4472), "Artifact lacks RayQueryKHR"); }
            // A requested Slang capability permits operations; unused features
            // need not survive as OpCapability declarations in emitted SPIR-V.
            else if (name == "spvGroupNonUniformBallot") { }
            else if (name == "spvRayTracingClusterAccelerationStructureNV") { require(capabilities.contains(5437), "Artifact lacks cluster capability"); }
            else { require(name == "SPV_NV_cluster_acceleration_structure", "Unknown reference capability"); }
        }
        const auto& provenance = manifest_.at("provenance").at("manifests");
        require(provenance.is_array() && !provenance.empty(), "Artifact has no sealed provenance manifests");
        for (const auto& item : provenance) {
            const auto source = resolve(item.at("file").get<std::string>());
            require(debug::debugSha256(readFile(source)) == item.at("sha256").get<std::string>(), "Artifact provenance manifest hash mismatch");
            provenanceHashes_.push_back({{"file", source.string()}, {"sha256", item.at("sha256")}});
        }
        requiresNoFiber_ = manifest_.at("requires_no_fiber").get<bool>();
        if (postprocess() == "none") {
            require(!requiresNoFiber_, "Prepared SSA artifacts must retain the General material feature set");
            const auto& seal = artifact.at("prepared").at("provenance_manifest");
            const auto source = resolve(seal.at("file").get<std::string>());
            require(debug::debugSha256(readFile(source)) == seal.at("sha256").get<std::string>(), "Prepared SSA producer manifest changed");
            require(std::any_of(provenanceHashes_.begin(), provenanceHashes_.end(), [&](const auto& item) {
                return item.at("file").template get<std::string>() == source.string() && item.at("sha256") == seal.at("sha256");
            }), "Prepared SSA producer seal is absent from the artifact provenance");
        }
    }

    bool targets(const render::SlangShaderDesc& desc) const
    {
        const auto& request = manifest_.at("reference_request");
        return desc.moduleName && desc.entryPointName && desc.moduleName == request.at("module").get<std::string>() &&
            desc.entryPointName == request.at("entry").get<std::string>();
    }

    std::optional<render::Result<render::ShaderArtifactOverrideResult>> resolve(
        const render::SlangShaderDesc& desc, std::string&) override
    {
        if (!targets(desc)) { return std::nullopt; }
        render::ShaderArtifactOverrideResult result;
        result.shader = resolveShader(desc);
        result.identity.moduleName = sourceTag();
        result.identity.sourceEntryPoint = sourceEntry();
        result.identity.entryPoint = "main";
        result.identity.descriptorHeapMode = "mapped";
        result.identity.optimizationLevel = "O1";
        result.identity.debugMode = "release";
        result.identity.spirvPostprocess = postprocess();
        result.identity.spirvPostprocessToolVersion = toolVersion();
        result.identity.inputSpirvHash = std::stoull(fnv1a64_, nullptr, 16);
        return render::Result<render::ShaderArtifactOverrideResult>(std::move(result));
    }

    render::ShaderCompileResult resolveShader(const render::SlangShaderDesc& desc)
    {
        try {
            require(targets(desc), "Resolver invoked for a different shader target");
            const auto& request = manifest_.at("reference_request");
            require(desc.profileName && desc.profileName == request.at("profile").get<std::string>() &&
                desc.searchPath && normalized(desc.searchPath) == normalized(request.at("search_path").get<std::string>()), "Artifact request profile/source root mismatch");
            std::vector<std::string> paths;
            for (const char* path : desc.additionalSearchPaths) { require(path != nullptr, "Null shader search path"); paths.push_back(normalized(path)); }
            std::vector<std::string> referencePaths;
            for (const auto& path : request.at("additional_search_paths")) { referencePaths.push_back(normalized(path.get<std::string>())); }
            require(paths == referencePaths, "Artifact additional search paths mismatch");
            std::set<std::string> capabilities;
            for (const char* cap : desc.capabilities) { require(cap != nullptr, "Null shader capability"); capabilities.insert(cap); }
            require(capabilities == request.at("capabilities").get<std::set<std::string>>() && capabilities.size() == desc.capabilities.size(), "Artifact request capabilities mismatch");
            std::map<std::string, std::string> defines;
            for (const auto& define : desc.macroDefines) {
                require(define.name && define.value && defines.emplace(define.name, define.value).second, "Invalid/duplicate shader macro");
            }
            std::map<std::string, std::string> referenceDefines;
            for (const auto& define : request.at("defines")) {
                require(referenceDefines.emplace(define.at(0).get<std::string>(), define.at(1).get<std::string>()).second, "Duplicate reference macro");
            }
            require(defines == referenceDefines && defines.at("METALLIC_CUSTOM_MATERIALS") == "0" &&
                !defines.contains("SHARC_UPDATE") && !defines.contains("SHARC_QUERY") && !defines.contains("METALLIC_PT_SINGLE_SAMPLE"),
                "Artifact request permutation/macros mismatch");
            require(render::slangShaderOptimizationLevel(desc) == render::SlangShaderOptimizationLevel::O1 &&
                !render::slangUsesNativeDescriptorHeap(desc) && desc.descriptorHeapStrides == render::DescriptorHeapShaderStrides{} &&
                render::slangShaderDebugMode() == render::SlangShaderDebugMode::Disabled &&
                render::slangShaderPostprocessMode(desc) == render::SlangShaderPostprocessMode::None &&
                render::sceneWorkingColorSpace() == render::SceneWorkingColorSpace::ACEScg, "Artifact effective compiler/descriptor/debug/color ABI mismatch");
            {
                std::lock_guard lock(evidenceMutex_);
                ++consumed_;
            }
            render::ShaderCompileResult result;
            result.spirv = words_;
            // No compile, disk-cache hit, or saved-cache metrics are fabricated.
            return result;
        } catch (const std::exception& error) {
            std::lock_guard lock(evidenceMutex_);
            ++rejected_;
            errors_.push_back(error.what());
            throw;
        }
    }

    Json inspectScene(const scene::SceneDocument& document, const Json& graph) const
    {
        require(!document.materials().empty(), "Artifact fixture needs an explicit loaded scene material set");
        inspectMaterials(document.materials(), requiresNoFiber_);
        size_t pathPasses = 0;
        for (const auto& node : graph.at("nodes")) {
            const auto type = node.at("type").get<std::string>();
            if (requiresNoFiber_) {
                require(type.find("Fiber") == std::string::npos && type.find("Strand") == std::string::npos &&
                    type.find("RTXCR") == std::string::npos, "NoFiber artifact rejected: graph can introduce Fiber geometry");
                if (node.at("properties").contains("path")) {
                    require(normalized(std::filesystem::path(PROJECT_SOURCE_DIR) /
                        node.at("properties").at("path").get<std::string>()) == normalized(document.sourcePath()),
                        "NoFiber artifact rejected: graph references another uninspected scene");
                }
            }
            if (type != "ScenePathTracePass") { continue; }
            ++pathPasses;
            const auto& properties = node.at("properties");
            require(properties.value("cacheMode", "sharc") == "off" && properties.value("samples", 1) == 2 &&
                properties.value("bsdf", "") == "openpbr", "Artifact graph requires SHARC off, OpenPBR and two samples per frame");
            require(properties.contains("path") && normalized(std::filesystem::path(PROJECT_SOURCE_DIR) /
                properties.at("path").get<std::string>()) == normalized(document.sourcePath()), "Graph path differs from the inspected scene");
        }
        require(pathPasses == 1, "Artifact fixture requires exactly one fixed scene path-trace pass");
        return {{"requires_no_fiber", requiresNoFiber_}, {"eligible", true}, {"materials_inspected", document.materials().size()},
            {"scene_sources_inspected", document.sources().size()}, {"scope", "Complete loaded material set; primary, secondary and shadow rays"},
            {"custom_material_programs", 0}, {"fiber_materials", requiresNoFiber_ ? Json(0) : Json(nullptr)}};
    }

    static void inspectMaterials(std::span<const scene::RenderMaterial> materials, bool noFiber)
    {
        for (const auto& material : materials) {
            require(material.valueProgram.empty(), "Artifact fixture does not support custom material programs");
            if (noFiber) { require(!material.rtxcrHair, "NoFiber artifact rejected: scene material set contains Fiber"); }
        }
    }

    Json capturedRecipes(const std::filesystem::path& directory) const
    {
        require(std::filesystem::is_directory(directory), "Artifact fixture requires actual pipeline recipe capture");
        Json recipes = Json::array();
        for (const auto& file : std::filesystem::directory_iterator(directory)) {
            if (!file.is_regular_file() || !file.path().filename().string().ends_with(".recipe.json")) { continue; }
            std::string diagnostics;
            auto recipe = render::readPipelineRecipe(file.path(), diagnostics);
            require(bool(recipe), "Cannot validate captured pipeline recipe: " + diagnostics);
            if (recipe->type != render::PipelineRecipeType::Compute || recipe->computeShader.moduleName != sourceTag()) { continue; }
            const auto& shader = recipe->computeShader;
            const auto bytes = std::span(reinterpret_cast<const uint8_t*>(shader.spirv.data()), shader.spirv.size() * sizeof(uint32_t));
            require(shader.spirv == words_ && debug::debugSha256(bytes) == sha256_ && hashHex(bytes) == fnv1a64_ &&
                shader.sourceEntryPoint == sourceEntry() && shader.entryPoint == "main" && shader.descriptorHeapMode == "mapped" &&
                shader.debugMode == "release" && shader.optimizationLevel == "O1" && shader.spirvPostprocess == postprocess() &&
                shader.spirvPostprocessToolVersion == toolVersion() && recipe->usesBindlessHeap && !recipe->indirectBindable &&
                recipe->bindlessUserPushDataSize == 12 && !recipe->disableOptimization &&
                recipe->computeBackend == render::ComputePipelineBackend::Pipeline,
                "Actual captured PSO did not consume the artifact/ABI");
            recipes.push_back({{"file", file.path().string()}, {"expected_pso_hash", recipe->expectedPsoHash},
                {"input_sha256", sha256_}, {"input_fnv1a64", fnv1a64_}, {"input_bytes", bytes.size()},
                {"source_tag", shader.moduleName}, {"source_identity_ambiguous", shader.sourceIdentityAmbiguous}});
        }
        require(!recipes.empty(), "No actual artifact PSO recipe captured; substitution cannot qualify");
        return recipes;
    }

    void verifyInputsUnchanged() const
    {
        require(debug::debugSha256(readFile(manifestPath_)) == manifestSha256_ &&
            debug::debugSha256(readFile(artifactPath_)) == sha256_, "Artifact manifest/bytecode changed during correctness run");
        for (const auto& source : provenanceHashes_) {
            require(debug::debugSha256(readFile(source.at("file").get<std::string>())) == source.at("sha256").get<std::string>(),
                "Artifact provenance changed during correctness run");
        }
    }

    Json evidence() const
    {
        std::lock_guard lock(evidenceMutex_);
        Json result = {{"artifact_id", manifest_.at("artifact_id")}, {"manifest", manifestPath_.string()}, {"manifest_sha256", manifestSha256_},
            {"artifact", artifactPath_.string()}, {"input_sha256", sha256_}, {"input_fnv1a64", fnv1a64_}, {"input_bytes", words_.size() * 4},
            {"source_tag", sourceTag()}, {"postprocess", postprocess()}, {"postprocess_tool_version", toolVersion()},
            {"statistics", statistics_}, {"provenance_manifests", provenanceHashes_}, {"consumed", consumed_}, {"rejected_target", rejected_},
            {"errors", errors_}, {"metrics_scope", "Precompiled artifact substitution; no compiler/cache timing"}};
        if (postprocess() == "none") { result["prepared"] = manifest_.at("artifact").at("prepared"); }
        return result;
    }

    std::string sourceTag() const { return manifest_.at("artifact").at("source_tag").get<std::string>(); }
    std::string postprocess() const { return manifest_.at("artifact").at("spirv_postprocess").get<std::string>(); }
    std::string toolVersion() const { return manifest_.at("artifact").at("tool_version").get<std::string>(); }
    std::string sourceEntry() const { return manifest_.at("reference_request").at("entry").get<std::string>(); }
    uint64_t consumed() const { std::lock_guard lock(evidenceMutex_); return consumed_; }
    uint64_t rejected() const { std::lock_guard lock(evidenceMutex_); return rejected_; }
    const std::string& inputHash() const { return fnv1a64_; }
    const std::string& inputSha256() const { return sha256_; }
    size_t inputBytes() const { return words_.size() * sizeof(uint32_t); }

    static void validatePostprocessProvenance(const Json& artifact)
    {
        const auto policy = artifact.at("spirv_postprocess").get<std::string>();
        const auto version = artifact.at("tool_version").get<std::string>();
        if (policy == "dead-branches") {
            require(!version.empty(), "Legacy cleaned artifact lacks its optimizer provenance");
            require(!artifact.contains("prepared"), "External SSA preparation cannot be labelled dead-branches");
            return;
        }
        require(policy == "none" && version.empty(), "Prepared artifacts require runtime postprocess none and no runtime optimizer version");
        const auto& prepared = artifact.at("prepared");
        const auto transform = prepared.at("transform").get<std::string>();
        require(transform == "external-cleaned" || transform == "external-ssa", "Unknown external artifact preparation");
        const Json ancestry = {"--eliminate-dead-branches", "--simplify-instructions"};
        const Json passes = transform == "external-ssa" ? Json::array({"--ssa-rewrite"}) : Json::array();
        require(prepared.at("ancestry_passes") == ancestry && prepared.at("passes") == passes,
            "Prepared artifact is not the exact two-pass ancestor plus the selected SSA-only step");
        const auto inputBytes = prepared.at("input_bytes").get<size_t>();
        const auto inputHash = prepared.at("input_sha256").get<std::string>();
        require(inputBytes >= 20 && inputBytes % sizeof(uint32_t) == 0 && inputBytes <= 64 * 1024 * 1024 && validSha256(inputHash),
            "Prepared SSA input identity is invalid");
        require(!prepared.at("tool_version").get<std::string>().empty(), "Prepared artifact lacks its external optimizer version");
        const auto& seal = prepared.at("provenance_manifest");
        require(!seal.at("file").get<std::string>().empty() && validSha256(seal.at("sha256").get<std::string>()),
            "Prepared artifact lacks its sealed producer provenance");
        if (transform == "external-cleaned") {
            require(artifact.at("bytes").get<size_t>() == inputBytes && artifact.at("sha256").get<std::string>() == inputHash,
                "The cleaned control must be exactly its prepared input bytes");
        }
    }

    static void require(bool condition, const std::string& message)
    {
        if (!condition) { throw std::runtime_error(message); }
    }

    static std::vector<uint8_t> readFile(const std::filesystem::path& path)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "Cannot open artifact input: " + path.string());
        const auto size = file.tellg();
        require(size >= 0 && size <= 64 * 1024 * 1024, "Artifact input exceeds bounded loader size");
        std::vector<uint8_t> bytes(static_cast<size_t>(size));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(bool(file), "Truncated artifact input: " + path.string());
        return bytes;
    }

private:
    static bool validSha256(std::string_view value)
    {
        return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char digit) {
            return (digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f');
        });
    }
    static std::string normalized(const std::filesystem::path& path)
    {
        return std::filesystem::weakly_canonical(std::filesystem::absolute(path)).generic_string();
    }
    std::filesystem::path resolve(const std::filesystem::path& path) const
    {
        return std::filesystem::absolute(path.is_absolute() ? path : manifestPath_.parent_path() / path).lexically_normal();
    }
    static std::string hashHex(std::span<const uint8_t> raw)
    {
        std::ostringstream text;
        text << std::hex << std::setw(16) << std::setfill('0') << render::detail::hashBytes(render::detail::kFnvOffset, raw.data(), raw.size());
        return text.str();
    }
    Json manifest_, statistics_, provenanceHashes_ = Json::array();
    std::filesystem::path manifestPath_, artifactPath_;
    std::string manifestSha256_, sha256_, fnv1a64_;
    std::vector<uint32_t> words_;
    bool requiresNoFiber_ = false;
    uint64_t consumed_ = 0, rejected_ = 0;
    mutable std::mutex evidenceMutex_;
    std::vector<std::string> errors_;
};

} // namespace metallic::tests
