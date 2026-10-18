#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/GAPI/Hash.h"

#include <json.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace metallic::render {
namespace {

using Json = nlohmann::json;
constexpr uint32_t kRecipeVersion = 4;
constexpr uint32_t kSpirvMagic = 0x07230203u;
constexpr uint64_t kMaxSpirvBytes = 256ull * 1024 * 1024;

std::string hexHash(uint64_t value)
{
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << value;
    return text.str();
}

uint64_t readHash(const Json& value)
{
    const auto text = value.get<std::string>();
    if (text.size() != 16 || text.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
        throw std::runtime_error("invalid 64-bit content hash");
    }
    return std::stoull(text, nullptr, 16);
}

uint64_t shaderHash(const PipelineRecipeShader& shader)
{
    return detail::hashBytes(detail::kFnvOffset, shader.spirv.data(), shader.spirv.size() * sizeof(uint32_t));
}

uint32_t readUint32(const Json& value)
{
    if (!value.is_number_integer()) { throw std::runtime_error("pipeline state must contain integer values"); }
    const auto number = value.get<uint64_t>();
    if (number > UINT32_MAX) { throw std::runtime_error("pipeline state integer exceeds 32 bits"); }
    return static_cast<uint32_t>(number);
}

template<typename T>
T readEnum(const Json& value, T largest)
{
    const auto number = readUint32(value);
    if (number > static_cast<uint32_t>(largest)) { throw std::runtime_error("unsupported pipeline state enum"); }
    return static_cast<T>(number);
}

Json deviceJson(const PipelineRecipeDevice& device)
{
    const auto& caps = device.capabilities;
    return {{"device_name", device.deviceName}, {"driver_name", device.driverName},
        {"vendor_id", device.vendorId}, {"device_id", device.deviceId}, {"driver_version", device.driverVersion},
        {"api_version", device.apiVersion}, {"pipeline_cache_uuid", device.pipelineCacheUUID},
        {"validation_enabled", device.validationEnabled},
        {"synchronization_validation_enabled", device.synchronizationValidationEnabled},
        {"enabled_features", {
            {"bindless_descriptor_heap", caps.bindlessDescriptorHeap}, {"shader_object", caps.shaderObject},
            {"mesh_shader", caps.meshShader}, {"task_shader", caps.taskShader},
            {"geometry_shader", caps.geometryShader}, {"task_shader_subgroup_ballot", caps.taskShaderSubgroupBallot},
            {"subgroup_size_control", caps.subgroupSizeControl}, {"compute_full_subgroups", caps.computeFullSubgroups},
            {"shader_maximal_reconvergence", caps.shaderMaximalReconvergence},
            {"ray_tracing_acceleration_structure", caps.rayTracingAccelerationStructure}, {"ray_query", caps.rayQuery},
            {"ray_tracing_pipeline", caps.rayTracingPipeline},
            {"ray_tracing_invocation_reorder", caps.rayTracingInvocationReorder},
            {"ray_tracing_invocation_reorder_reordering_hint", caps.rayTracingInvocationReorderReorderingHint},
            {"cluster_acceleration_structure", caps.clusterAccelerationStructure},
            {"partitioned_acceleration_structure", caps.partitionedAccelerationStructure},
            {"device_generated_commands", caps.deviceGeneratedCommands}}}};
}

PipelineRecipeDevice readDevice(const Json& json)
{
    PipelineRecipeDevice device;
    device.deviceName = json.at("device_name").get<std::string>();
    device.driverName = json.at("driver_name").get<std::string>();
    device.vendorId = readUint32(json.at("vendor_id"));
    device.deviceId = readUint32(json.at("device_id"));
    device.driverVersion = readUint32(json.at("driver_version"));
    device.apiVersion = readUint32(json.at("api_version"));
    device.pipelineCacheUUID = json.at("pipeline_cache_uuid").get<std::string>();
    device.validationEnabled = json.at("validation_enabled").get<bool>();
    device.synchronizationValidationEnabled = json.at("synchronization_validation_enabled").get<bool>();
    const auto& features = json.at("enabled_features");
    auto& caps = device.capabilities;
    caps.bindlessDescriptorHeap = features.at("bindless_descriptor_heap").get<bool>();
    caps.shaderObject = features.at("shader_object").get<bool>();
    caps.meshShader = features.at("mesh_shader").get<bool>();
    caps.taskShader = features.at("task_shader").get<bool>();
    caps.geometryShader = features.at("geometry_shader").get<bool>();
    caps.taskShaderSubgroupBallot = features.at("task_shader_subgroup_ballot").get<bool>();
    caps.subgroupSizeControl = features.at("subgroup_size_control").get<bool>();
    caps.computeFullSubgroups = features.at("compute_full_subgroups").get<bool>();
    caps.shaderMaximalReconvergence = features.value("shader_maximal_reconvergence", false);
    caps.rayTracingAccelerationStructure = features.at("ray_tracing_acceleration_structure").get<bool>();
    caps.rayQuery = features.at("ray_query").get<bool>();
    caps.rayTracingPipeline = features.value("ray_tracing_pipeline", false);
    caps.rayTracingInvocationReorder = features.value("ray_tracing_invocation_reorder", false);
    caps.rayTracingInvocationReorderReorderingHint = features.value("ray_tracing_invocation_reorder_reordering_hint", false);
    caps.clusterAccelerationStructure = features.at("cluster_acceleration_structure").get<bool>();
    caps.partitionedAccelerationStructure = features.at("partitioned_acceleration_structure").get<bool>();
    caps.deviceGeneratedCommands = features.at("device_generated_commands").get<bool>();
    return device;
}

void validateShader(const PipelineRecipeShader& shader)
{
    if (shader.spirv.size() < 5 || shader.spirv.size() * sizeof(uint32_t) > kMaxSpirvBytes ||
        shader.spirv[0] != kSpirvMagic || shader.entryPoint.empty() || shader.entryPoint.find('\0') != std::string::npos) {
        throw std::runtime_error("invalid recipe shader bytecode or entry point");
    }
    const auto canonical = detail::canonicalShaderSpecializationConstants(shader.specializationConstants);
    if (canonical != shader.specializationConstants) { throw std::runtime_error("recipe specialization constants must be canonical"); }
    std::string specializationError;
    if (!detail::validateShaderSpecializationModule(shader.spirv, canonical, specializationError)) {
        throw std::runtime_error(specializationError);
    }
    if (shader.inputSpirvHash && shader.inputSpirvHash != shaderHash(shader)) {
        throw std::runtime_error("recipe SPIR-V content hash mismatch");
    }
    if (shader.spirvPostprocess != "none" && shader.spirvPostprocess != "dead-branches" &&
        shader.spirvPostprocess != "slang-dead-branches") {
        throw std::runtime_error("invalid recipe SPIR-V postprocess policy");
    }
    if ((shader.spirvPostprocess == "dead-branches" || shader.spirvPostprocess == "slang-dead-branches") &&
        (shader.spirvPostprocessToolVersion.empty() || shader.spirvPostprocessToolVersion == "unavailable")) {
        throw std::runtime_error("recipe SPIR-V cleanup tool version is missing or unavailable");
    }
}

void validateRecipe(const PipelineRecipe& recipe)
{
    if (recipe.type != PipelineRecipeType::Compute && recipe.type != PipelineRecipeType::Graphics &&
        recipe.type != PipelineRecipeType::RayTracing) {
        throw std::runtime_error("unsupported recipe pipeline type");
    }
    for (const auto* shader : {&recipe.computeShader, &recipe.vertexShader, &recipe.taskShader, &recipe.meshShader,
        &recipe.fragmentShader, &recipe.rayGenerationShader, &recipe.missShader, &recipe.closestHitShader, &recipe.anyHitShader}) {
        if (shader->spirv.empty() && !shader->specializationConstants.empty()) {
            throw std::runtime_error("specialization values require a shader stage");
        }
    }
    if (recipe.colorAttachmentCount > recipe.colorFormats.size()) { throw std::runtime_error("too many color attachments"); }
    if (recipe.computeBackend > ComputePipelineBackend::PipelineBinary ||
        (recipe.type != PipelineRecipeType::Compute && recipe.computeBackend != ComputePipelineBackend::Pipeline) ||
        (recipe.computeBackend == ComputePipelineBackend::ShaderObject && recipe.disableOptimization)) {
        throw std::runtime_error("invalid recipe compute backend or optimization policy");
    }
    const bool hasRayTracingStages = !recipe.rayGenerationShader.spirv.empty() || !recipe.missShader.spirv.empty() ||
        !recipe.closestHitShader.spirv.empty() || !recipe.anyHitShader.spirv.empty();
    if (recipe.type != PipelineRecipeType::RayTracing && hasRayTracingStages) {
        throw std::runtime_error("non-ray-tracing recipe contains ray-tracing stages");
    }
    if (recipe.enableInvocationReorder && (recipe.type != PipelineRecipeType::RayTracing ||
        !recipe.sourceDevice.capabilities.rayTracingInvocationReorder || !recipe.sourceDevice.capabilities.rayTracingPipeline)) {
        throw std::runtime_error("invocation reorder requires an EXT-enabled ray-tracing source device");
    }
    if (recipe.type == PipelineRecipeType::RayTracing) {
        if (!recipe.computeShader.spirv.empty() || !recipe.vertexShader.spirv.empty() || !recipe.taskShader.spirv.empty() ||
            !recipe.meshShader.spirv.empty() || !recipe.fragmentShader.spirv.empty() || recipe.indirectBindable ||
            recipe.maxRecursionDepth == 0) {
            throw std::runtime_error("invalid ray-tracing recipe stage combination or state");
        }
        for (const auto* shader : {&recipe.rayGenerationShader, &recipe.missShader, &recipe.closestHitShader, &recipe.anyHitShader}) {
            validateShader(*shader);
        }
        if (!recipe.missShader.specializationConstants.empty() || !recipe.closestHitShader.specializationConstants.empty() ||
            !recipe.anyHitShader.specializationConstants.empty()) {
            throw std::runtime_error("ray-tracing specialization is supported on ray generation only");
        }
    } else if (recipe.type == PipelineRecipeType::Compute) {
        validateShader(recipe.computeShader);
        if (!recipe.vertexShader.spirv.empty() || !recipe.taskShader.spirv.empty() ||
            !recipe.meshShader.spirv.empty() || !recipe.fragmentShader.spirv.empty()) {
            throw std::runtime_error("compute recipe contains graphics stages");
        }
    } else {
        const bool vertex = !recipe.vertexShader.spirv.empty();
        const bool mesh = !recipe.meshShader.spirv.empty();
        if (vertex == mesh || (!recipe.taskShader.spirv.empty() && !mesh) || !recipe.computeShader.spirv.empty()) {
            throw std::runtime_error("invalid graphics recipe stage combination");
        }
        for (const auto* shader : {&recipe.vertexShader, &recipe.taskShader, &recipe.meshShader, &recipe.fragmentShader}) {
            if (!shader->specializationConstants.empty()) { throw std::runtime_error("graphics specialization is outside this compute-only API"); }
            if (!shader->spirv.empty()) { validateShader(*shader); }
        }
        if (recipe.taskRequireFullSubgroups && !recipe.taskRequiredSubgroupSize) {
            throw std::runtime_error("full task subgroups require a fixed subgroup size");
        }
    }
}

Json writeShader(const PipelineRecipeShader& shader, const std::filesystem::path& directory, uint32_t version)
{
    if (shader.spirv.empty()) { return {{"spirv_file", nullptr}, {"entry_point", shader.entryPoint}}; }
    validateShader(shader);
    const std::string hash = hexHash(shaderHash(shader));
    const auto name = hash + ".spv";
    const auto path = directory / name;
    // Identical modules share one input blob; refuse a content-address collision
    // rather than replacing the bytecode used by an already-exported recipe.
    if (std::filesystem::exists(path)) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() != static_cast<std::streamoff>(shader.spirv.size() * sizeof(uint32_t))) {
            throw std::runtime_error("existing SPIR-V blob has an unexpected size");
        }
        std::vector<uint32_t> words(shader.spirv.size());
        input.seekg(0);
        input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
        if (!input || words != shader.spirv) { throw std::runtime_error("existing SPIR-V blob differs from its content hash"); }
    } else {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(shader.spirv.data()), static_cast<std::streamsize>(shader.spirv.size() * sizeof(uint32_t)));
        output.close();
        if (!output) { throw std::runtime_error("cannot write recipe SPIR-V blob"); }
    }
    Json json{{"spirv_file", name}, {"spirv_bytes", shader.spirv.size() * sizeof(uint32_t)},
        {"input_spirv_hash", hash}, {"module_name", shader.moduleName},
        {"source_entry_point", shader.sourceEntryPoint}, {"entry_point", shader.entryPoint},
        {"debug_mode", shader.debugMode}, {"descriptor_heap_mode", shader.descriptorHeapMode},
        {"slang_optimization", shader.optimizationLevel}, {"source_identity_ambiguous", shader.sourceIdentityAmbiguous},
        {"spirv_postprocess", shader.spirvPostprocess},
        {"spirv_postprocess_tool_version", shader.spirvPostprocessToolVersion}};
    if (version >= 2) {
        json["specialization_constants"] = Json::array();
        for (const auto& value : shader.specializationConstants) {
            json["specialization_constants"].push_back({{"id", value.id},
                {"type", value.type == ShaderSpecializationType::Int32 ? "int32" : "bool32"},
                {"size", 4}, {"bits", value.bits}});
        }
    }
    return json;
}

PipelineRecipeShader readShader(const Json& json, const std::filesystem::path& directory, uint32_t version)
{
    PipelineRecipeShader shader;
    if (json.at("spirv_file").is_null()) {
        if (json.contains("specialization_constants")) { throw std::runtime_error("empty stage cannot carry specialization data"); }
        shader.entryPoint = json.at("entry_point").get<std::string>();
        return shader;
    }
    const auto name = json.at("spirv_file").get<std::string>();
    const std::filesystem::path relative(name);
    if (relative.empty() || relative.is_absolute() || relative.has_parent_path() || relative.filename() != relative) {
        throw std::runtime_error("recipe SPIR-V files must be sibling filenames");
    }
    const auto byteSize = json.at("spirv_bytes").get<uint64_t>();
    if (byteSize < 5 * sizeof(uint32_t) || byteSize > kMaxSpirvBytes || byteSize % sizeof(uint32_t)) {
        throw std::runtime_error("invalid recipe SPIR-V size");
    }
    std::ifstream input(directory / relative, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(byteSize)) { throw std::runtime_error("recipe SPIR-V file size mismatch"); }
    shader.spirv.resize(static_cast<size_t>(byteSize / sizeof(uint32_t)));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(shader.spirv.data()), static_cast<std::streamsize>(byteSize));
    if (!input) { throw std::runtime_error("cannot read recipe SPIR-V file"); }
    if (version >= 2) {
        const auto& values = json.at("specialization_constants");
        if (!values.is_array() || values.size() > 1024) { throw std::runtime_error("invalid specialization array"); }
        for (const auto& value : values) {
            const auto type = value.at("type").get<std::string>();
            if ((type != "int32" && type != "bool32") || readUint32(value.at("size")) != 4) {
                throw std::runtime_error("invalid scalar32 specialization type or size");
            }
            shader.specializationConstants.push_back({readUint32(value.at("id")),
                type == "int32" ? ShaderSpecializationType::Int32 : ShaderSpecializationType::Bool32,
                readUint32(value.at("bits"))});
        }
    } else if (json.contains("specialization_constants")) {
        throw std::runtime_error("legacy recipe cannot carry specialization values");
    }
    shader.inputSpirvHash = readHash(json.at("input_spirv_hash"));
    shader.moduleName = json.at("module_name").get<std::string>();
    shader.sourceEntryPoint = json.at("source_entry_point").get<std::string>();
    shader.entryPoint = json.at("entry_point").get<std::string>();
    shader.debugMode = json.at("debug_mode").get<std::string>();
    shader.descriptorHeapMode = json.at("descriptor_heap_mode").get<std::string>();
    shader.optimizationLevel = json.value("slang_optimization", std::string{});
    shader.spirvPostprocess = json.value("spirv_postprocess", std::string{"none"});
    shader.spirvPostprocessToolVersion = json.value("spirv_postprocess_tool_version", std::string{});
    shader.sourceIdentityAmbiguous = json.value("source_identity_ambiguous", false);
    if (shader.inputSpirvHash != shaderHash(shader)) { throw std::runtime_error("recipe SPIR-V content hash mismatch"); }
    validateShader(shader);
    return shader;
}

} // namespace

ComputePipelineDesc PipelineRecipe::computeDesc(ShaderModule* module, PipelineCache* cache) const
{
    return {.computeShader = {module, computeShader.entryPoint.c_str(), computeShader.specializationConstants}, .usesBindlessHeap = usesBindlessHeap,
        .bindlessUserPushDataSize = bindlessUserPushDataSize, .pipelineCache = cache, .indirectBindable = indirectBindable,
        .disableOptimization = disableOptimization, .backend = computeBackend};
}

GraphicsPipelineDesc PipelineRecipe::graphicsDesc(ShaderModule* vertex, ShaderModule* task,
    ShaderModule* mesh, ShaderModule* fragment, PipelineCache* cache) const
{
    return {.vertexShader = {vertex, vertexShader.entryPoint.c_str()},
        .taskShader = {task, taskShader.entryPoint.c_str()}, .meshShader = {mesh, meshShader.entryPoint.c_str()},
        .fragmentShader = {fragment, fragmentShader.entryPoint.c_str()},
        .taskRequiredSubgroupSize = taskRequiredSubgroupSize, .taskRequireFullSubgroups = taskRequireFullSubgroups,
        .colorFormats = colorFormats, .colorAttachmentCount = colorAttachmentCount,
        .depthStencilFormat = depthStencilFormat, .topology = topology, .rasterization = rasterization,
        .depthStencil = depthStencil, .usesBindlessHeap = usesBindlessHeap,
        .pipelineCache = cache, .indirectBindable = indirectBindable, .disableOptimization = disableOptimization};
}

RayTracingPipelineDesc PipelineRecipe::rayTracingDesc(ShaderModule* rayGeneration, ShaderModule* miss,
    ShaderModule* closestHit, ShaderModule* anyHit, PipelineCache* cache) const
{
    return {.rayGenerationShader = {rayGeneration, rayGenerationShader.entryPoint.c_str(), rayGenerationShader.specializationConstants},
        .missShader = {miss, missShader.entryPoint.c_str(), missShader.specializationConstants},
        .closestHitShader = {closestHit, closestHitShader.entryPoint.c_str(), closestHitShader.specializationConstants},
        .anyHitShader = {anyHit, anyHitShader.entryPoint.c_str(), anyHitShader.specializationConstants},
        .usesBindlessHeap = usesBindlessHeap, .bindlessUserPushDataSize = bindlessUserPushDataSize,
        .pipelineCache = cache, .maxRecursionDepth = maxRecursionDepth,
        .clusterAccelerationStructure = clusterAccelerationStructure, .enableInvocationReorder = enableInvocationReorder,
        .disableOptimization = disableOptimization};
}

Result<> writePipelineRecipe(const PipelineRecipe& recipe, const std::filesystem::path& path, std::string& log)
{
    std::filesystem::path temporary;
    try {
        validateRecipe(recipe);
        const auto directory = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
        std::filesystem::create_directories(directory);
        // Preserve version-1 empty-stage output and old cache identities. A
        // specialized compute capture uses v2 with a required explicit vector.
        const uint32_t version = recipe.type == PipelineRecipeType::RayTracing ? (recipe.enableInvocationReorder ? kRecipeVersion : 3) :
            recipe.computeShader.specializationConstants.empty() ? 1 : 2;
        Json json{{"schema", "metallic_pipeline_recipe"}, {"version", version},
            {"pipeline_type", recipe.type == PipelineRecipeType::RayTracing ? "ray_tracing" :
                recipe.type == PipelineRecipeType::Compute ? "compute" : "graphics"},
            {"expected_pso_hash", hexHash(recipe.expectedPsoHash)}, {"cache_group", recipe.cacheGroup},
            {"source_device", deviceJson(recipe.sourceDevice)},
            {"shaders", {{"compute", writeShader(recipe.computeShader, directory, version)},
                {"vertex", writeShader(recipe.vertexShader, directory, version)}, {"task", writeShader(recipe.taskShader, directory, version)},
                {"mesh", writeShader(recipe.meshShader, directory, version)}, {"fragment", writeShader(recipe.fragmentShader, directory, version)}}},
            {"state", {{"uses_bindless_heap", recipe.usesBindlessHeap}, {"indirect_bindable", recipe.indirectBindable},
                {"disable_optimization", recipe.disableOptimization}, {"compute_backend", static_cast<uint32_t>(recipe.computeBackend)},
                {"bindless_user_push_data_size", recipe.bindlessUserPushDataSize},
                {"task_required_subgroup_size", recipe.taskRequiredSubgroupSize},
                {"task_require_full_subgroups", recipe.taskRequireFullSubgroups},
                {"color_attachment_count", recipe.colorAttachmentCount}, {"depth_stencil_format", static_cast<uint32_t>(recipe.depthStencilFormat)},
                {"topology", static_cast<uint32_t>(recipe.topology)},
                {"cull_mode", static_cast<uint32_t>(recipe.rasterization.cullMode)},
                {"front_face", static_cast<uint32_t>(recipe.rasterization.frontFace)},
                {"depth_test_enable", recipe.depthStencil.depthTestEnable}, {"depth_write_enable", recipe.depthStencil.depthWriteEnable},
                {"depth_compare_op", static_cast<uint32_t>(recipe.depthStencil.depthCompareOp)}}}};
        for (const auto format : recipe.colorFormats) { json["state"]["color_formats"].push_back(static_cast<uint32_t>(format)); }
        if (recipe.type == PipelineRecipeType::RayTracing) {
            json["shaders"]["ray_generation"] = writeShader(recipe.rayGenerationShader, directory, version);
            json["shaders"]["miss"] = writeShader(recipe.missShader, directory, version);
            json["shaders"]["closest_hit"] = writeShader(recipe.closestHitShader, directory, version);
            json["shaders"]["any_hit"] = writeShader(recipe.anyHitShader, directory, version);
            json["state"]["max_recursion_depth"] = recipe.maxRecursionDepth;
            json["state"]["cluster_acceleration_structure"] = recipe.clusterAccelerationStructure;
            if (version >= 4) { json["state"]["enable_invocation_reorder"] = recipe.enableInvocationReorder; }
        }
        static std::atomic<uint64_t> sequence{0};
        temporary = path;
        temporary += ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence.fetch_add(1));
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output << json.dump(2) << '\n';
        output.close();
        if (!output) { throw std::runtime_error("cannot write recipe manifest"); }
        // A PSO identity can recur with different validation/device settings.
        // Publish the current provenance rather than retaining a stale capture.
        // POSIX replaces atomically; Windows requires removal before rename.
        std::error_code renameError;
        std::filesystem::rename(temporary, path, renameError);
        if (renameError) {
            if (!std::filesystem::exists(path)) { throw std::filesystem::filesystem_error("cannot publish recipe manifest", temporary, path, renameError); }
            std::filesystem::remove(path);
            std::filesystem::rename(temporary, path);
        }
        return {};
    } catch (const std::exception& error) {
        if (!temporary.empty()) { std::error_code ignored; std::filesystem::remove(temporary, ignored); }
        log += "Pipeline recipe export failed: " + std::string(error.what()) + '\n';
        return makeError(Error::Failure);
    }
}

Result<PipelineRecipe> readPipelineRecipe(const std::filesystem::path& path, std::string& log)
{
    try {
        std::ifstream input(path, std::ios::binary);
        if (!input) { throw std::runtime_error("cannot open recipe manifest"); }
        const auto json = Json::parse(input);
        const uint32_t version = readUint32(json.at("version"));
        if (json.at("schema") != "metallic_pipeline_recipe" || (version < 1 || version > kRecipeVersion)) {
            throw std::runtime_error("unsupported pipeline recipe schema/version");
        }
        PipelineRecipe recipe;
        const auto type = json.at("pipeline_type").get<std::string>();
        if ((type != "compute" && type != "graphics" && type != "ray_tracing") ||
            ((type == "ray_tracing") != (version >= 3))) {
            throw std::runtime_error("unsupported recipe pipeline type/version");
        }
        recipe.type = type == "ray_tracing" ? PipelineRecipeType::RayTracing :
            type == "compute" ? PipelineRecipeType::Compute : PipelineRecipeType::Graphics;
        recipe.expectedPsoHash = readHash(json.at("expected_pso_hash"));
        recipe.cacheGroup = json.at("cache_group").get<std::string>();
        recipe.sourceDevice = readDevice(json.at("source_device"));
        const auto directory = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
        const auto& shaders = json.at("shaders");
        recipe.computeShader = readShader(shaders.at("compute"), directory, version);
        recipe.vertexShader = readShader(shaders.at("vertex"), directory, version);
        recipe.taskShader = readShader(shaders.at("task"), directory, version);
        recipe.meshShader = readShader(shaders.at("mesh"), directory, version);
        recipe.fragmentShader = readShader(shaders.at("fragment"), directory, version);
        const auto& state = json.at("state");
        if (recipe.type == PipelineRecipeType::RayTracing) {
            recipe.rayGenerationShader = readShader(shaders.at("ray_generation"), directory, version);
            recipe.missShader = readShader(shaders.at("miss"), directory, version);
            recipe.closestHitShader = readShader(shaders.at("closest_hit"), directory, version);
            recipe.anyHitShader = readShader(shaders.at("any_hit"), directory, version);
            recipe.maxRecursionDepth = readUint32(state.at("max_recursion_depth"));
            recipe.clusterAccelerationStructure = state.at("cluster_acceleration_structure").get<bool>();
            recipe.enableInvocationReorder = version >= 4 ? state.at("enable_invocation_reorder").get<bool>() : false;
        } else if (shaders.contains("ray_generation") || shaders.contains("miss") ||
            shaders.contains("closest_hit") || shaders.contains("any_hit")) {
            throw std::runtime_error("ray-tracing stages require recipe version 3 or later");
        }
        if (version < 4 && state.contains("enable_invocation_reorder")) {
            throw std::runtime_error("invocation reorder state requires recipe version 4");
        }
        recipe.usesBindlessHeap = state.at("uses_bindless_heap").get<bool>();
        recipe.indirectBindable = state.at("indirect_bindable").get<bool>();
        recipe.disableOptimization = state.value("disable_optimization", false);
        recipe.computeBackend = state.contains("compute_backend")
            ? readEnum(state.at("compute_backend"), ComputePipelineBackend::PipelineBinary) : ComputePipelineBackend::Pipeline;
        recipe.bindlessUserPushDataSize = readUint32(state.at("bindless_user_push_data_size"));
        recipe.taskRequiredSubgroupSize = readUint32(state.at("task_required_subgroup_size"));
        recipe.taskRequireFullSubgroups = state.at("task_require_full_subgroups").get<bool>();
        recipe.colorAttachmentCount = readUint32(state.at("color_attachment_count"));
        const auto& formats = state.at("color_formats");
        if (formats.size() != recipe.colorFormats.size()) { throw std::runtime_error("invalid color format array size"); }
        for (size_t i = 0; i < recipe.colorFormats.size(); ++i) { recipe.colorFormats[i] = readEnum(formats[i], Format::BC7sRGB); }
        recipe.depthStencilFormat = readEnum(state.at("depth_stencil_format"), Format::BC7sRGB);
        recipe.topology = readEnum(state.at("topology"), PrimitiveTopology::TriangleList);
        recipe.rasterization.cullMode = readEnum(state.at("cull_mode"), CullMode::Back);
        recipe.rasterization.frontFace = readEnum(state.at("front_face"), FrontFace::Clockwise);
        recipe.depthStencil.depthTestEnable = state.at("depth_test_enable").get<bool>();
        recipe.depthStencil.depthWriteEnable = state.at("depth_write_enable").get<bool>();
        recipe.depthStencil.depthCompareOp = readEnum(state.at("depth_compare_op"), CompareOp::Always);
        validateRecipe(recipe);
        return recipe;
    } catch (const std::exception& error) {
        log += "Pipeline recipe import failed: " + std::string(error.what()) + '\n';
        return makeError(Error::InvalidArgument);
    }
}

} // namespace metallic::render
