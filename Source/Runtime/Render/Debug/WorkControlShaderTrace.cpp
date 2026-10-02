#include "WorkControlShaderTrace.h"
#include "Runtime/Render/Core/StreamRasterParameters.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include <slang.h>
#include <fstream>
#include <map>
#include <cstdlib>
#include <stdexcept>

namespace metallic::render {
using debug::DebugValue;
namespace {
void require(bool valid, const char* message)
{
    if (!valid) { throw std::runtime_error(message); }
}
void save(const std::filesystem::path& path, const DebugValue& value)
{
    std::ofstream file(path);
    file << debug::encodeLossless(value).dump(2) << '\n';
    require(bool(file), "Cannot save shader trace evidence");
}
std::string hashFile(const std::filesystem::path& path)
{
    std::ifstream file(path,std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
    require(bool(file) && !bytes.empty(),"Cannot hash shader dependency");
    return debug::debugSha256(bytes);
}
DebugValue sourceHashes(const std::vector<std::string>& paths)
{
    DebugValue hashes = DebugValue::object();
    for (const auto& path : paths) { hashes[path] = hashFile(path); }
    return hashes;
}
void validateSources(const DebugValue& hashes)
{
    for (auto item = hashes.begin(); item != hashes.end(); ++item) {
        require(hashFile(item.key()) == item.value().get<std::string>(),"Shader source changed during observation");
    }
}
std::span<const uint8_t> bytes(const std::vector<uint32_t>& spirv)
{
    return {reinterpret_cast<const uint8_t*>(spirv.data()),spirv.size()*4};
}
} // namespace

struct WorkControlShaderTrace::Lease {
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::atomic<int> submission{0};
};

WorkControlShaderTrace::WorkControlShaderTrace(debug::DebugCore& core, const std::filesystem::path& output)
    : core_(core), runtime_(core), output_(std::filesystem::absolute(output) / "shader-trace")
{
    require(std::filesystem::create_directory(output_),"Shader trace output must be new");
    const auto settingsDirectory = output_ / "layer-settings";
    require(std::filesystem::create_directory(settingsDirectory),"Layer settings output exists");
    std::ofstream file(settingsDirectory / "vk_layer_settings.txt");
    file << "khronos_validation.printf_only_preset = false\n"
         << "khronos_validation.printf_enable = true\n"
         << "khronos_validation.printf_to_stdout = false\n"
         << "khronos_validation.printf_verbose = false\n"
         << "khronos_validation.printf_buffer_size = 65536\n"
         << "khronos_validation.report_flags = info,warn,error\n"
         << "khronos_validation.enable_message_limit = false\n"
         << "khronos_validation.duplicate_message_limit = 0\n"
         << "khronos_validation.message_id_filter = \n";
    file.close(); require(bool(file),"Cannot write isolated layer settings");
    if (const char* previous = std::getenv("VK_LAYER_SETTINGS_PATH")) { previousSettings_ = previous; }
#ifdef _WIN32
    require(_putenv_s("VK_LAYER_SETTINGS_PATH",settingsDirectory.string().c_str()) == 0,"Cannot set process layer settings");
#else
    require(setenv("VK_LAYER_SETTINGS_PATH",settingsDirectory.string().c_str(),1) == 0,"Cannot set process layer settings");
#endif
}

WorkControlShaderTrace::~WorkControlShaderTrace()
{
#ifdef _WIN32
    _putenv_s("VK_LAYER_SETTINGS_PATH",previousSettings_ ? previousSettings_->c_str() : "");
#else
    if (previousSettings_) { setenv("VK_LAYER_SETTINGS_PATH",previousSettings_->c_str(),1); }
    else { unsetenv("VK_LAYER_SETTINGS_PATH"); }
#endif
}

void WorkControlShaderTrace::qualify(Device& device, Queue& queue, const std::filesystem::path& output)
{
    require(output_ == std::filesystem::absolute(output) / "shader-trace","Shader trace output changed");
    require(slangShaderDebugMode() == SlangShaderDebugMode::Disabled,"P2 requires normal optimization/debug mode");
    ShaderCompileResult code;
    const auto compiled = compileSlangShaderToSpirv({.moduleName="Features/Debug/ShaderTraceBackend",.entryPointName="echoMain",
        .searchPath=PROJECT_SOURCE_DIR "/Shaders"}, {.enableDiskCache=false}, code.diagnostics).transform([&](auto value) { code = std::move(value); });
    require(bool(compiled),code.diagnostics.c_str());
    auto shader = device.createShaderModule({.spirv = code.spirv});
    require(bool(shader),"Backend echo shader creation failed");
    auto pipeline = device.createComputePipeline({.computeShader = {shader->get(), "main"}});
    require(bool(pipeline),"Backend echo pipeline creation failed");
    auto pool = device.createCommandPool(queue); require(bool(pool),"Echo command pool failed");
    auto commands = (*pool)->createCommandBuffer(); require(bool(commands),"Echo commands failed");
    QueueSubmissionTracker tracker; require(bool(tracker.initialize(device,queue)),"Echo tracker failed");
    RenderFrameContext frame; require(bool(frame.begin(0)),"Echo frame failed");
    require(bool((*commands)->begin(&frame)),"Echo recording failed");
    if (auto commandResult = (*commands)->bindExecution((*pipeline)->execution()); !commandResult) { throw std::runtime_error(std::string("bindExecution failed: ") + metallic::render::resultToString(commandResult)); } (*commands)->dispatch(1,1,1);
    require(bool((*commands)->end()),"Echo recording end failed");
    CommandBuffer* submitted[]{commands->get()};
    require(bool(tracker.submit({.commandBuffers = {submitted, 1}},frame)),"Echo submit failed");
    if (!frame.wait(10'000'000'000ull)) { std::_Exit(2); }
    require(bool(queue.waitIdle()),"Echo queue drain failed");
    size_t echoes = 0;
    DebugValue raw = DebugValue::array();
    bool healthy = capture_.dropped() == 0 && capture_.truncated() == 0;
    for (const auto& message : capture_.drain()) {
        raw.push_back({{"id",message.id},{"severity",message.severity},{"text",message.text.data()}});
        healthy &= (message.severity & (256|4096)) == 0;
        if (message.id == 0x4fe1fef9 && message.severity == 16 &&
            std::string_view(message.text.data()).find("MTQ1 backend-ready") != std::string_view::npos) { ++echoes; }
    }
    const auto native = vulkan::nativeDevice(device);
    VkPhysicalDeviceProperties properties{}; vkGetPhysicalDeviceProperties(native.physicalDevice,&properties);
    slang::IGlobalSession* session = nullptr;
    require(SLANG_SUCCEEDED(slang::createGlobalSession(&session)),"Slang version unavailable");
    evidence_ = {{"backend","VVL.DebugPrintf"},{"collectionBoundary","case-process-instance-destroyed"},
        {"slang",session->getBuildTagString()},{"gpu",properties.deviceName},{"driverVersionRaw",properties.driverVersion},
        {"settingsTransport","isolated process layer-settings file"},
        {"settingsFileSha256",hashFile(output_/"layer-settings/vk_layer_settings.txt")},
        {"streamline",device.capabilities().streamline},
        {"layerSpecVersion",capture_.layerSpecVersion},{"bufferBytes",capture_.options().bufferBytes},
        {"backendEchoCount",echoes},{"backendEchoRaw",raw},{"performanceEligible",false}};
    session->release(); save(output_/"Backend.json",evidence_);
    require(healthy && echoes == 1,"Production-instance shader printf echo failed");
    require(bool((*pool)->reset()) && bool(frame.reset()),"Echo cleanup failed");
}

void WorkControlShaderTrace::configure(DebugValue graph)
{
    require(!armed_,"Cannot reconfigure an armed observation");
    graph_ = std::move(graph); core_.setGraph(graph_);
    const auto hashes = sourceHashes({PROJECT_SOURCE_DIR "/Shaders/Features/GPUDriven/GPUDrivenStreamWorkRaster.slang",
        PROJECT_SOURCE_DIR "/Shaders/Modules/ShaderTrace.slang",
        PROJECT_SOURCE_DIR "/Shaders/Modules/GPUDriven/HybridRasterTriangle.slang"});
    DebugValue fields = DebugValue::array();
    for (const char* name : {"recordIndex","triangleId","instanceFlags","triangleCount"}) { fields.push_back({{"name",name},{"type","u32"}}); }
    for (const char* vertex : {"a","b","c"}) {
        fields.push_back({{"name",std::string(vertex)+"PositionX"},{"type","i32"}});
        fields.push_back({{"name",std::string(vertex)+"PositionY"},{"type","i32"}});
        fields.push_back({{"name",std::string(vertex)+"Depth"},{"type","f32"}});
    }
    sites_ = DebugValue::array();
    for (const char* phase : {"early","late"}) {
        sites_.push_back(debug::shaderTraceSite({{"name","stream.after-triangle-prepare"},{"id",2},{"phase",phase},
            {"adapter","work-control-v1"},{"pass","VBuffer"},{"module","Features/GPUDriven/GPUDrivenStreamWorkRaster"},
            {"entry","streamClusterRasterWorkControlMain"},{"sourceHashes",hashes},{"fields",fields},
            {"invocation",{{"group",{0,0,0}},{"localIndex",0}}}}));
    }
    DebugValue decisions = DebugValue::array();
    for (const auto& [name,type] : std::array<std::pair<const char*,const char*>,12>{{
        {"recordIndex","u32"},{"triangleId","u32"},{"instanceFlags","u32"},{"signedArea","i32"},
        {"doubleSided","u32"},{"reason","u32"},{"lowerX","i32"},{"lowerY","i32"},
        {"upperX","i32"},{"upperY","i32"},{"determinant","f32"},{"evaluatedStages","u32"}}}) {
        decisions.push_back({{"name",name},{"type",type}});
    }
    for (const char* phase : {"early","late"}) {
        sites_.push_back(debug::shaderTraceSite({{"name","stream.triangle-decision"},{"id",3},{"phase",phase},
            {"adapter","work-control-v1"},{"pass","VBuffer"},{"module","Features/GPUDriven/GPUDrivenStreamWorkRaster"},
            {"entry","streamClusterRasterWorkControlMain"},{"sourceHashes",hashes},{"fields",decisions},
            {"decisionReasons",{{"0","SetupAccepted"},{"1","DegenerateArea"},{"2","Backface"},
                {"3","EmptyBounds"},{"4","DegenerateDepthPlane"}}},
            {"evaluatedStages",{{"area",1},{"bounds",2},{"plane",4}}},
            {"invocation",{{"group",{0,0,0}},{"localIndex",0}}}}));
    }
    core_.configureShaderTrace({{"configured",true},{"smokeVerified",true},{"fixtureOnly",false},
        {"adapter","work-control-v1"},{"collectionBoundary","case-process-instance-destroyed"},
        {"scope","one watch in a dedicated primed workload process"}},sites_);
    save(output_/"Sites.json",sites_);
}

const DebugValue& WorkControlShaderTrace::site(std::string_view name, std::string_view phase) const
{
    for (const auto& item : sites_) {
        if (item.at("name").get_ref<const std::string&>() == name && item.at("phase").get_ref<const std::string&>() == phase) { return item; }
    }
    throw std::runtime_error("Unknown WorkControl site/phase");
}

void WorkControlShaderTrace::prepare(Device& device, const DebugValue& specification, const DebugValue& input)
{
    require(job_.empty(),"P2 permits one observation per process");
    request_ = specification; phase_ = request_.at("target").at("phase");
    auto response = core_.dispatch({{"method","shader.watch"},{"params",request_}});
    require(response.at("status") == "ok",response.dump().c_str());
    job_ = response.at("result").at("job");
    auto requests = core_.takeShaderRequests(graph_.at("id").get<std::string>(),debug::debugUnsigned(graph_.at("generation")));
    require(requests.size()==1,"Shader request not available");
    const auto& site = this->site(request_.at("target").at("site").get<std::string>(),phase_);
    auto begun = runtime_.begin(requests[0],site,{{"graph",graph_.at("id")},{"generation",graph_.at("generation")},
        {"execution",0u},{"pass","VBuffer"},{"phase",phase_},{"dispatchOrdinal",phase_ == "early" ? 0 : 1},
        {"inputFingerprint",debug::debugSha256(input.dump())}});
    require(bool(begun),"Cannot begin shader observation");
    evidence_["input"] = input;
    validateSources(site.at("sourceHashes"));
    std::map<std::string,std::string> defines{{"METALLIC_WORK_CONTROL_TRACE","1"},
        {"TRACE_SITE_ID",std::to_string(debug::debugUnsigned(site.at("id")))},
        {"TRACE_LOCAL",std::to_string(debug::debugUnsigned(request_.at("invocation").at("localIndex")))},
        {"TRACE_MAX_RECORDS",std::to_string(debug::debugUnsigned(request_.at("limits").at("maxRecords")))},
        {"TRACE_PREDICATE_ENABLED",request_.contains("predicate") ? "1" : "0"},
        {"TRACE_TRIANGLE_ID",request_.contains("predicate") ? std::to_string(debug::debugUnsigned(request_["predicate"]["value"])) : "0"}};
    for (const auto& [token,macro] : std::array<std::pair<const char*,const char*>,3>{{{"sessionToken","SESSION"},{"runToken","RUN"},{"dispatchToken","DISPATCH"}}}) {
        const auto value = debug::debugUnsigned(runtime_.plan().at(token));
        defines[std::string("TRACE_")+macro+"_LO"] = std::to_string(uint32_t(value));
        defines[std::string("TRACE_")+macro+"_HI"] = std::to_string(uint32_t(value>>32));
    }
    const char* axes[]{"X","Y","Z"};
    for (size_t i=0;i<3;++i) { defines[std::string("TRACE_GROUP_")+axes[i]] = std::to_string(debug::debugUnsigned(request_["invocation"]["group"][i])); }
    std::vector<SlangMacroDefine> macros;
    for (const auto& [name,value] : defines) { macros.push_back({name.c_str(),value.c_str()}); }
    ShaderCompileResult code;
    const auto compiled = compileSlangShaderToSpirv({
        .moduleName = "Features/GPUDriven/GPUDrivenStreamWorkRaster",
        .entryPointName = "streamClusterRasterWorkControlMain",
        .searchPath = PROJECT_SOURCE_DIR "/Shaders",
        .macroDefines = macros,
    }, {.enableDiskCache=false}, code.diagnostics).transform([&](auto value) { code = std::move(value); });
    if (!compiled) { runtime_.stop("CompileFailed"); evidence_["diagnostics"]=code.diagnostics; throw std::runtime_error(code.diagnostics); }
    DebugValue variant{{"instrumentation","Printf"},{"module",site.at("module")},{"entry",site.at("entry")},
        {"compilerSpirvSha256",debug::debugSha256(bytes(code.spirv))},{"compilerSpirvHex",debug::hexEncode(bytes(code.spirv))},
        {"deviceSpirv",nullptr},{"macros",defines},{"dependencyHashes",sourceHashes(code.dependencies)},
        {"descriptorHeapMode",std::getenv("METALLIC_SLANG_DESCRIPTOR_MODE") ? std::getenv("METALLIC_SLANG_DESCRIPTOR_MODE") : "mapped"},
        {"profile","spirv_1_6"},{"shaderDebugMode","Disabled"},{"diskCache",false},{"slang",evidence_.at("slang")}};
    validateSources(site.at("sourceHashes")); runtime_.compiledVariant(variant); evidence_["variant"]=variant;
    lease_ = std::make_shared<Lease>();
    require(bool(device.createShaderModule({.spirv = code.spirv}).transform(
        [&](auto value){lease_->shader=std::move(value);})),"Diagnostic shader creation failed");
    require(bool(device.createComputePipeline({
        .computeShader = {lease_->shader.get(), "main"},
        .usesBindlessHeap = true,
        .bindlessUserPushDataSize = sizeof(StreamRasterParameters),
    }).transform(
        [&](auto value){lease_->pipeline=std::move(value);})),"Diagnostic pipeline creation failed");
    armed_ = true;
}

void WorkControlShaderTrace::beginExecution(debug::DebugEvidenceStamp stamp)
{
    execution_ = std::move(stamp);
}

bool WorkControlShaderTrace::bind(CommandBuffer& commands, std::string_view pass, const DebugValue& production)
{
    if (!armed_ || bound_ || pass != "VBuffer" || production.value("phase", "") != phase_) { return false; }
    require(execution_.graph == graph_.at("id").get<std::string>() && execution_.generation == debug::debugUnsigned(graph_.at("generation")),"Stale graph at shader bind");
    require(production.at("mode") == 5 && production.at("queue") == "graphics" && production.at("snapshotFrozen") == true &&
        production.at("entryPoint") == "streamClusterRasterWorkControlMain","Unsupported production shader branch");
    validateSources(evidence_.at("variant").at("dependencyHashes"));
    if (!runtime_.maySubmit()) { armed_=false; return false; }
    require(commands.frameContext()!=nullptr,"Untracked shader dispatch");
    runtime_.recorded({{"execution",execution_.execution},{"frameSlot",execution_.frameSlot},
        {"commandBufferRecording",commands.frameContext()->frameIndex()}});
    completion_ = commands.frameContext()->completion();
    auto lease = lease_;
    require(bool(commands.retainResource(lease)),"Cannot retain diagnostic pipeline");
    require(bool(commands.addSubmissionTransaction(std::make_shared<SubmissionTransaction>(
        [lease]{lease->submission.store(1);},[lease]{lease->submission.store(-1);}))),"Cannot track diagnostic submit");
    require(bool(commands.bindExecution(lease->pipeline->execution())), "Cannot bind diagnostic pipeline");
    evidence_["productionBinding"] = production;
    evidence_["execution"] = execution_.value();
    bound_=true; armed_=false;
    return true;
}

void WorkControlShaderTrace::targetDrained(Queue& queue)
{
    require(bound_ && lease_ && lease_->submission.load()==1 && completion_.isSubmitted() && completion_.isComplete(),
        "Target dispatch did not complete");
    std::vector<SemaphoreSubmitDesc> signals;
    require(bool(completion_.appendWaits(signals)) && !signals.empty(),"Missing tracked frame completion signals");
    DebugValue values = DebugValue::array();
    for (const auto& signal : signals) { values.push_back(signal.value); }
    runtime_.submitted({{"type","Graphics"},{"family",vulkan::nativeQueue(queue).familyIndex}},
        {{"timelineValue",signals.size()==1 ? DebugValue(signals.front().value) : DebugValue(nullptr)},
         {"frameTimelineValues",values},{"execution",execution_.execution},{"scope","tracked frame completion"}});
    gpuComplete_ = true;
    evidence_["targetGpuComplete"] = true;
    completion_ = {};
}

void WorkControlShaderTrace::restoration(DebugValue evidence, bool readbackValid)
{
    readbackValid_ = readbackValid;
    evidence_["restoration"] = std::move(evidence);
}
void WorkControlShaderTrace::abort(std::string reason)
{
    armed_=false;
    if (!job_.empty()) { runtime_.stop("ExecutionFailed"); evidence_["error"]=std::move(reason); }
}
void WorkControlShaderTrace::releaseGpu()
{
    armed_=false; completion_={}; lease_.reset();
}
void WorkControlShaderTrace::finishAfterDevice() noexcept
{
    if (job_.empty()) { return; }
    try {
        runtime_.finish(capture_,gpuComplete_,true,readbackValid_,evidence_);
        const auto state = core_.dispatch({{"method","jobs.get"},{"params",{{"job",job_},{"includeCapture",true}}}});
        save(output_/"Job.json",state);
        require(state.at("result").contains("artifactCount"),"No sealed shader artifact");
        const auto manifest = state.at("result").at("capture");
        const auto directory = output_/"capture"; require(std::filesystem::create_directory(directory),"Capture output exists");
        for (size_t i=0;i<manifest.at("artifacts").size();++i) {
            std::ofstream file(directory/(std::to_string(i)+".bin"),std::ios::binary);
            uint64_t offset=0, total=debug::debugUnsigned(manifest["artifacts"][i]["bytes"]);
            while (offset<total) {
                auto response=core_.dispatch({{"method","artifact.read"},{"params",{{"job",job_},{"index",i},{"offset",offset}}}});
                require(response.at("status")=="ok","Artifact read failed");
                const auto data=debug::hexDecode(response.at("result").at("hex").get<std::string>());
                require(bool(data)&&!data->empty(),"Empty artifact chunk");
                file.write(reinterpret_cast<const char*>(data->data()),data->size()); offset+=data->size();
            }
            require(bool(file),"Artifact write failed");
        }
        save(directory/"manifest.json",manifest);
    } catch (const std::exception& error) {
        try { save(output_/"Failure.json",{{"error",error.what()}}); } catch (...) {}
    }
}
} // namespace metallic::render
