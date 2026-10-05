#include "MaterialShaderWarmupRequests.h"
#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Scene/scene.h"

#include <json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

namespace metallic::tools {
namespace {

nlohmann::json readJson(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file) { throw std::runtime_error("Cannot read shader warmup input: " + path.string()); }
    return nlohmann::json::parse(file);
}

std::filesystem::path resolve(const std::filesystem::path& root, const std::string& path)
{
    const std::filesystem::path input(path);
    return input.is_absolute() ? input : root / input;
}

nlohmann::json installedCatalog(const std::filesystem::path& root, const std::filesystem::path& path, const char* prefix)
{
    // Match the runtime catalog's availability checks. Stale optional samples
    // must not prevent an unrelated built-in scene from starting.
    try {
        const auto catalog = readJson(path);
        if (catalog.at("version") != 1 || !catalog.at("samples").is_array() || catalog.at("samples").size() > 64) {
            return nlohmann::json::array();
        }
        std::set<std::string> ids;
        for (const auto& sample : catalog.at("samples")) {
            const auto id = sample.at("id").get<std::string>();
            for (const char* key : {"name", "description", "environment"}) {
                if (!sample.at(key).is_string()) { return nlohmann::json::array(); }
            }
            if (!id.starts_with(prefix) || !ids.insert(id).second ||
                !std::filesystem::is_regular_file(resolve(root, sample.at("scenePath").get<std::string>())) ||
                !std::filesystem::is_regular_file(resolve(root, sample.at("graphPath").get<std::string>()))) {
                return nlohmann::json::array();
            }
        }
        return catalog.at("samples");
    } catch (const std::exception& error) {
        spdlog::warn("[ShaderWarmup] Ignoring unavailable local catalog '{}': {}", path.string(), error.what());
        return nlohmann::json::array();
    }
}

} // namespace

std::vector<render::ShaderRequest> materialShaderWarmupRequests(
    const std::filesystem::path& projectRoot, const std::string& rtxcrInclude)
{
    using namespace render;
    std::vector<ShaderRequest> requests;
    std::map<std::filesystem::path, std::shared_ptr<const MaterialValueProgramSet>> programs;
    const auto add = [&](ShaderRequest request) {
        if (std::find(requests.begin(), requests.end(), request) == requests.end()) {
            requests.push_back(std::move(request));
        }
    };
    for (const auto& [catalogPath, prefix] : {
            std::pair{"build/MaterialValidation/PainterLookDev/Catalog.json", "painter-"},
            std::pair{"build/MaterialValidation/WhiteStudio02/Catalog.json", "studio-white-"}}) {
        const auto path = projectRoot / catalogPath;
        if (!std::filesystem::exists(path)) { continue; }
        const auto samples = installedCatalog(projectRoot, path, prefix);
        for (const auto& sample : samples) {
            const auto scenePath = resolve(projectRoot, sample.at("scenePath").get<std::string>());
            auto sidecar = scenePath;
            sidecar.replace_extension(".metallic_scene.json");
            if (!std::filesystem::exists(sidecar)) { continue; }
            auto& set = programs[sidecar];
            if (!set) {
                std::vector<scene::RenderMaterial> materials;
                for (const auto& entry : readJson(sidecar).value("materials", nlohmann::json::array())) {
                    scene::RenderMaterial candidate;
                    std::string diagnostics;
                    if (entry.contains("materialAsset")) {
                        const auto& binding = entry.at("materialAsset");
                        material::MaterialAssetLibrary library(resolve(sidecar.parent_path(), binding.at("root").get<std::string>()));
                        material::ResolvedMaterialInstance instance;
                        if (!library.resolve(binding.at("uri").get<std::string>(), instance, diagnostics) ||
                            !material::lowerMaterialInstance(instance, [](std::string_view) { return 0; }, candidate, diagnostics)) {
                            throw std::runtime_error(sidecar.string() + ": " + diagnostics);
                        }
                    }
                    // SceneDocument applies local overrides after the asset.
                    const auto properties = entry.value("properties", nlohmann::json::object());
                    candidate.valueProgram = properties.value("valueProgram", candidate.valueProgram);
                    candidate.alphaMode = properties.value("alphaMode", candidate.alphaMode);
                    if (properties.contains("valueParameters")) {
                        candidate.valueParameters = properties.at("valueParameters").get<std::array<float, 16>>();
                    }
                    materials.push_back(std::move(candidate));
                }
                std::string diagnostics;
                set = MaterialValueProgramSet::create(materials, diagnostics);
                if (!set) { throw std::runtime_error(sidecar.string() + ": " + diagnostics); }
            }
            if (set->programCount() == 0) { continue; }
            std::filesystem::path include;
            std::string diagnostics;
            if (!set->writeInclude(projectRoot / ".cache/materials", include, diagnostics)) {
                throw std::runtime_error(diagnostics);
            }
            const auto graphPath = resolve(projectRoot, sample.at("graphPath").get<std::string>());
            const auto graph = readJson(graphPath);
            for (const auto& node : graph.at("nodes")) {
                const auto type = node.value("type", "");
                if (type != "ScenePathTracePass" && type != "VisibilityBufferDeferredPass") { continue; }
                const auto properties = node.value("properties", nlohmann::json::object());
                const bool deferred = type == "VisibilityBufferDeferredPass";
                const bool guides = properties.value("exportUpscalerGuides", false);
                const bool binned = deferred && properties.value("materialBinning", true) && properties.value("programBinning", true);
                const auto program = deferred ? (binned ? SceneShaderProgram::DeferredBinned : SceneShaderProgram::Deferred)
                    : properties.value("bsdf", "standard") == "openpbr"
                    ? (guides ? SceneShaderProgram::OpenPBRPathTraceGuides : SceneShaderProgram::OpenPBRPathTrace)
                    : (guides ? SceneShaderProgram::PathTraceGuides : SceneShaderProgram::PathTrace);
                for (bool fetch : {false, true}) {
                    if (deferred && fetch) { continue; }
                    for (bool view : {false, true}) {
                        if (!deferred && view) { continue; }
                        SceneShaderOptions options{.customMaterials = true, .globalView = view,
                            .hasRTXCR = !rtxcrInclude.empty(), .positionFetch = fetch,
                            .deferredFloat16 = properties.value("halfPrecision", true), .upscalerGuides = guides,
                            .materialInclude = include.string(), .rtxcrInclude = rtxcrInclude};
                        if (!binned) { add(makeSceneShaderRequest(program, options)); continue; }
                        for (int index = 0; index < 5; ++index) {
                            const auto value = std::to_string(index);
                            const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
                            add(makeSceneShaderRequest(program, options, {&define, 1}));
                        }
                        for (const auto& manifest : set->manifests()) {
                            if (manifest.closureFamily == MaterialClosureFamily::OpenPBRCompositeClosure) { continue; }
                            const auto id = std::to_string(manifest.programId);
                            const SlangMacroDefine defines[] = {{"MATERIAL_CLASS", "4"}, {"MATERIAL_PROGRAM_ID", id.c_str()}};
                            add(makeSceneShaderRequest(program, options, defines));
                        }
                    }
                }
            }
        }
    }
    return requests;
}

} // namespace metallic::tools
