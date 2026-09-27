#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanShaderPrintf.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Debug/ShaderTraceRuntime.h"
#include "Runtime/Debug/DebugTransport.h"
#include "Runtime/Debug/DebugHash.h"

#include <SDL3/SDL.h>
#include <slang.h>
#include <json.hpp>
#include <array>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
namespace render = metallic::render;
namespace vk = render::vulkan;
using Json = nlohmann::json;
namespace debug = metallic::debug;

void save(const std::filesystem::path& path, const Json& value)
{
    std::ofstream file(path);
    file << value.dump(2) << '\n';
    if (!file) { throw std::runtime_error("Cannot write " + path.string()); }
}

template<typename T>
T require(render::Result<T> result, const char* action)
{
    if (!result) { throw std::runtime_error(std::string(action) + ": " + render::resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

std::string version(uint32_t value)
{
    return std::to_string(VK_API_VERSION_MAJOR(value)) + "." +
        std::to_string(VK_API_VERSION_MINOR(value)) + "." + std::to_string(VK_API_VERSION_PATCH(value));
}

std::string loadedModule(const wchar_t* name)
{
#ifdef _WIN32
    const HMODULE module = GetModuleHandleW(name);
    std::array<wchar_t, 32768> path{};
    const DWORD length = module ? GetModuleFileNameW(module, path.data(), DWORD(path.size())) : 0;
    if (length && length < path.size()) { return std::filesystem::path(path.data()).generic_string(); }
#endif
    return {};
}

// Check the actual OpExtInst, rather than accepting an unused import or a text substring.
uint32_t printfInstructions(const std::vector<uint32_t>& words)
{
    std::set<uint32_t> imports;
    uint32_t count = 0;
    bool mainEntry = false;
    if (words.size() < 5 || words[0] != 0x07230203) { throw std::runtime_error("Invalid SPIR-V header"); }
    for (size_t i = 5; i < words.size();) {
        const uint32_t size = words[i] >> 16, opcode = words[i] & 0xffff;
        if (!size || size > words.size() - i) { throw std::runtime_error("Malformed SPIR-V"); }
        if (opcode == 15 && size >= 5 && words[i + 1] == 5 &&
            std::memcmp(&words[i + 3], "main", 5) == 0) { mainEntry = true; }
        if (opcode == 11 && size >= 3) {
            const char* text = reinterpret_cast<const char*>(&words[i + 2]);
            const size_t bytes = (size - 2) * 4;
            constexpr char name[] = "NonSemantic.DebugPrintf";
            if (bytes >= sizeof(name) && std::memcmp(text, name, sizeof(name)) == 0) { imports.insert(words[i + 1]); }
        }
        if (opcode == 12 && size >= 5 && imports.contains(words[i + 3]) && words[i + 4] == 1) { ++count; }
        i += size;
    }
    if (!mainEntry) { throw std::runtime_error("Expected compute SPIR-V entry point main"); }
    return count;
}

std::string sourceHash(const std::filesystem::path& path);

void run(Json& report, vk::ShaderPrintf& capture, const std::filesystem::path& directory,
    const std::string& mode, const std::string& fault, render::ShaderTraceRuntime* trace = nullptr, const Json& watch = Json::object())
{
    const auto validateSources = [&] {
        if (!trace) { return; }
        const auto& sources = trace->plan().at("sourceHashes");
        for (auto source = sources.begin(); source != sources.end(); ++source) {
            if (sourceHash(source.key()) != source.value().get<std::string>()) {
                trace->stop("StaleHandle");
                throw std::runtime_error("Shader source changed; restart fixture service");
            }
        }
    };
    validateSources();
    const bool heapMode = mode != "ordinary";
    const uint32_t recordCount = fault == "gpu-overflow" ? 128 : 1;
    report["expectedRecords"] = recordCount;
    const char* entry = heapMode ? "heapMain" : "ordinaryMain";
    const std::string macroValue = std::to_string(recordCount);
    std::vector<render::SlangMacroDefine> macros{{"PRINTF_RECORD_COUNT", macroValue.c_str()}};
    std::array<std::string, 8> traceValues;
    if (trace) {
        const auto& plan = trace->plan();
        const uint64_t session = debug::debugUnsigned(plan.at("sessionToken"));
        const uint64_t run = debug::debugUnsigned(plan.at("runToken"));
        const uint64_t dispatch = debug::debugUnsigned(plan.at("dispatchToken"));
        const std::string scenario = watch.value("fixtureScenario", "matched");
        const uint32_t scenarioId = scenario == "no-match" ? 1 : scenario == "site-not-reached" ? 2 : scenario == "missing-end" ? 3 : scenario == "quota" ? 4 : 0;
        const std::array<uint32_t,8> values{uint32_t(session),uint32_t(session>>32),uint32_t(run),uint32_t(run>>32),
            uint32_t(dispatch),uint32_t(dispatch>>32),uint32_t(debug::debugUnsigned(watch.at("limits").at("maxRecords"))),scenarioId};
        const char* names[]{"TRACE_SESSION_LO","TRACE_SESSION_HI","TRACE_RUN_LO","TRACE_RUN_HI","TRACE_DISPATCH_LO","TRACE_DISPATCH_HI","TRACE_MAX_RECORDS","TRACE_SCENARIO"};
        for (size_t i=0; i<values.size(); ++i) { traceValues[i]=std::to_string(values[i]); macros.push_back({names[i],traceValues[i].c_str()}); }
    }
    slang::IGlobalSession* slang = nullptr;
    if (SLANG_FAILED(slang::createGlobalSession(&slang))) { throw std::runtime_error("Slang session failed"); }
    report["slang"] = {{"buildTag", slang->getBuildTagString()},
        {"module", loadedModule(L"slang-compiler.dll")}, {"apiModule", loadedModule(L"slang.dll")}};
    slang->release();
    render::ShaderCompileResult compiled;
    render::setSlangShaderDebugMode(render::SlangShaderDebugMode::Disabled);
    const auto result = render::compileSlangShaderToSpirv({
        .moduleName = trace ? "ShaderTraceFixture" : "ShaderPrintfEcho",
        .entryPointName = entry,
        .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
        .macroDefines = macros,
        .descriptorHeapMode = mode == "heap-native" ? render::SlangDescriptorHeapMode::Native : render::SlangDescriptorHeapMode::Mapped,
    }, {.enableDiskCache = false}, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); });
    report["compiler"] = {{"profile", "spirv_1_6"}, {"diskCache", false}, {"shaderDebugMode", "Disabled"},
        {"diagnostics", compiled.diagnostics}, {"dependencies", compiled.dependencies}, {"sourceEntryPoint", entry}, {"spirvEntryPoint", "main"},
        {"macro", {{"PRINTF_RECORD_COUNT", macroValue}}}};
    if (!result && trace) { trace->stop("CompileFailed"); }
    require(result, "compile");
    report["compiler"]["debugPrintfInstructions"] = printfInstructions(compiled.spirv);
    if (trace) {
        const auto bytes = std::span(reinterpret_cast<const uint8_t*>(compiled.spirv.data()), compiled.spirv.size()*4);
        report["variant"] = {{"instrumentation", "Printf"}, {"descriptorMode", mode},
            {"compilerSpirvSha256", debug::debugSha256(bytes)}, {"compilerSpirvHex", debug::hexEncode(bytes)},
            {"deviceSpirv", nullptr}, {"compileOptions", report["compiler"]}, {"compiler", report["slang"]}, {"macros", Json::object()}};
        for (const auto& macro : macros) { report["variant"]["macros"][macro.name] = macro.value; }
        for (const auto& dependency : compiled.dependencies) {
            std::ifstream file(dependency, std::ios::binary);
            const std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
            if (!file || source.empty()) { throw std::runtime_error("Cannot read shader dependency"); }
            report["variant"]["dependencyHashes"][dependency] = debug::debugSha256(source);
        }
        validateSources();
        trace->compiledVariant(report["variant"]);
        if (!trace->maySubmit()) { throw std::runtime_error("Shader observation cancelled or stale before device creation"); }
    }
    if (printfInstructions(compiled.spirv) == 0) { throw std::runtime_error("No DebugPrintf instructions emitted"); }
    std::ofstream binary(directory / "compiler.spv", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(compiled.spirv.data()), compiled.spirv.size() * sizeof(uint32_t));
    binary.close();
    if (!binary) { throw std::runtime_error("Cannot write compiler.spv"); }
    report["phase"] = "device-create";
    save(directory / "Report.json", report);
    auto device = require(render::createDevice({.applicationName = "Metallic Shader Printf P0",
        .enableBindlessDescriptorHeap = heapMode, .shaderPrintf = &capture}), "createDevice");
    const auto native = vk::nativeDevice(*device);
    VkPhysicalDeviceDriverProperties driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
    vkGetPhysicalDeviceProperties2(native.physicalDevice, &properties);
    report["device"] = {{"name", properties.properties.deviceName}, {"apiVersion", version(properties.properties.apiVersion)},
        {"driverVersionRaw", properties.properties.driverVersion}, {"driverName", driver.driverName},
        {"driverInfo", driver.driverInfo}, {"vendorId", properties.properties.vendorID}, {"deviceId", properties.properties.deviceID},
        {"descriptorHeap", native.descriptorHeapEnabled}};
    report["loadedLayerModule"] = loadedModule(L"VkLayer_khronos_validation.dll");
    auto shader = require(device->createShaderModule({
        .spirv = compiled.spirv,
    }), "createShaderModule");
    struct Push { uint32_t inputBuffer; uint32_t cookie; };
    Push push{0, 305397763};
    report["phase"] = "pipeline-create";
    save(directory / "Report.json", report);
    auto pipeline = require(device->createComputePipeline({
        .computeShader = {shader.get(), "main"},
        .usesBindlessHeap = heapMode,
        .bindlessUserPushDataSize = heapMode ? sizeof(Push) : 0,
    }), "createComputePipeline");
    std::unique_ptr<render::BindlessHeap> heap;
    std::unique_ptr<render::Buffer> input, output;
    render::BindlessHandle inputHandle{}, outputHandle{};
    if (heapMode) {
        heap = require(device->createBindlessHeap({.maxSampledImages = 7, .maxBuffers = 4}), "createHeap");
        (void)require(heap->allocateBuffer(), "reserveUnusedSlot");
        inputHandle = require(heap->allocateBuffer(), "allocateInput");
        outputHandle = require(heap->allocateBuffer(), "allocateOutput");
        if (inputHandle.index == 0 || inputHandle.shaderIndex == inputHandle.index) {
            throw std::runtime_error("Probe must use nonzero slot and final descriptor indices");
        }
        input = require(device->createBuffer({.size = 16, .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::HostUpload}), "createInput");
        output = require(device->createBuffer({.size = 16, .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::HostReadback}), "createOutput");
        require(heap->writeStorageBuffer(inputHandle, *input), "writeInputDescriptor");
        require(heap->writeStorageBuffer(outputHandle, *output), "writeOutputDescriptor");
        const std::array<uint32_t, 4> values{outputHandle.shaderIndex, 73, 0, 0};
        void* mapped = input->map();
        if (!mapped) { throw std::runtime_error("Input map failed"); }
        std::memcpy(mapped, values.data(), sizeof(values));
        input->flush();
        input->unmap();
        push.inputBuffer = inputHandle.shaderIndex;
    }
    auto& queue = *device->getQueue(render::QueueType::Graphics);
    render::QueueSubmissionTracker tracker;
    require(tracker.initialize(*device, queue), "initializeTracker");
    auto pool = require(device->createCommandPool(queue), "createCommandPool");
    auto commands = require(pool->createCommandBuffer(), "createCommandBuffer");
    render::RenderFrameContext frame;
    require(frame.begin(trace ? debug::debugUnsigned(trace->plan().at("execution")) : 0), "beginFrame");
    require(commands->begin(&frame), "beginCommands");
    if (heapMode) {
        commands->bindBindlessHeap(*heap);
        if (auto commandResult = commands->bindExecution((pipeline)->execution(), &push, sizeof(push)); !commandResult) { throw std::runtime_error(std::string("bindExecution failed: ") + metallic::render::resultToString(commandResult)); }
    } else {
        if (auto commandResult = commands->bindExecution((pipeline)->execution()); !commandResult) { throw std::runtime_error(std::string("bindExecution failed: ") + metallic::render::resultToString(commandResult)); }
    }
    commands->dispatch(2, 1, 1);
    if (heapMode) {
        VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
        VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
        vkCmdPipelineBarrier2(vk::nativeCommandBuffer(*commands), &dependency);
    }
    require(commands->end(), "endCommands");
    report["phase"] = "submit";
    save(directory / "Report.json", report);
    render::CommandBuffer* submitted[] = {commands.get()};
    validateSources();
    if (trace && !trace->maySubmit()) { throw std::runtime_error("Shader observation cancelled or stale before submit"); }
    require(tracker.submit({.commandBuffers = {submitted, 1}}, frame), "submit");
    if (trace) { trace->submitted({{"type","Graphics"}, {"family",vk::nativeQueue(queue).familyIndex}},
        {{"timelineValue",frame.completion().value()}, {"execution",trace->plan().at("execution")}}); }
    if (!frame.wait(10'000'000'000ull)) {
        report["status"] = "gpu-wait-failed";
        save(directory / "Report.json", report);
        // Do not destroy resources still in flight after a timed-out wait.
        std::_Exit(2);
    }
    require(queue.waitIdle(), "waitForPrintfDelivery");
    report["gpuCompleted"] = true;
    if (heapMode) {
        std::array<uint32_t, 4> actual{};
        const void* mapped = output->map();
        if (!mapped) { throw std::runtime_error("Output map failed"); }
        output->invalidate();
        std::memcpy(actual.data(), mapped, sizeof(actual));
        output->unmap();
        const std::array<uint32_t, 4> expected{74, push.cookie, inputHandle.shaderIndex, outputHandle.shaderIndex};
        report["readback"] = {{"actual", actual}, {"expected", expected}, {"passed", actual == expected}};
        if (actual != expected) { throw std::runtime_error("Descriptor heap readback mismatch"); }
    }
    require(pool->reset(), "resetPool");
    require(frame.reset(), "resetFrame");
}

std::string sourceHash(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
    if (!file || bytes.empty()) { throw std::runtime_error("Cannot hash source " + path.string()); }
    return debug::debugSha256(bytes);
}

int serve(const std::filesystem::path& directory, const std::string& mode, uint32_t seconds)
{
    if (mode == "ordinary") { throw std::runtime_error("P1 service requires heap-mapped or heap-native"); }
    if (!SDL_Init(SDL_INIT_VIDEO)) { throw std::runtime_error(SDL_GetError()); }
    struct SdlCleanup { ~SdlCleanup() { SDL_Quit(); } } sdl;
    Json baseline;
    auto smoke = std::make_unique<vk::ShaderPrintf>();
    std::filesystem::create_directory(directory / "backend-smoke");
    run(baseline, *smoke, directory / "backend-smoke", mode, "none");
    size_t matches = 0;
    bool healthy = smoke->dropped() == 0 && smoke->truncated() == 0;
    for (const auto& m : smoke->snapshot()) {
        healthy &= (m.severity & (256 | 4096)) == 0;
        if (m.id == 0x4fe1fef9 && std::string_view(m.text.data()).find("MTP0 heap seq=0 group=1 lane=3 value=73 cookie=305397763") != std::string_view::npos) { ++matches; }
    }
    if (!healthy || matches != 1 || !baseline.value("gpuCompleted", false) || !baseline["readback"]["passed"].get<bool>()) {
        throw std::runtime_error("P1 backend live echo/readback failed");
    }
    debug::DebugCore core;
    core.setGraph({{"id","shader-trace-fixture"}, {"generation",1}, {"state","Ready"}});
    core.setEngineState({{"state","Ready"}, {"kind","ShaderTraceFixture"}});
    const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
    const Json sources{{(root/"tests/rhi/shaders/ShaderTraceFixture.slang").generic_string(), sourceHash(root/"tests/rhi/shaders/ShaderTraceFixture.slang")},
        {(root/"Shaders/Modules/ShaderTrace.slang").generic_string(),sourceHash(root/"Shaders/Modules/ShaderTrace.slang")}};
    const Json site = debug::shaderTraceSite({{"name","fixture.echo"}, {"id",1}, {"fixture",true},
        {"module","ShaderTraceFixture"}, {"entry","heapMain"}, {"sourceHashes",sources},
        {"descriptorMode",mode}, {"invocation",{{"group",{1,0,0}}, {"localIndex",3}}},
        {"fields", {{{"name","negativeZero"},{"type","f32"}}, {{"name","nanPayload"},{"type","f32"}},
            {{"name","large"},{"type","u64"}}, {{"name","value"},{"type","u32"}}}}});
    core.configureShaderTrace({{"configured",true}, {"smokeVerified",true}, {"backend","VVL.DebugPrintf"},
        {"scope",mode}, {"fixtureOnly",true}, {"limits",{{"activeJobs",1},{"targetFrames",1},{"maxRecords",16}, {"timeoutMs",30000}}},
        {"collectionBoundary","per-observation-instance-destroyed"}, {"runtime",baseline}}, Json::array({site}));
    render::ShaderTraceRuntime runtime(core);
    debug::DebugServer server(core);
    const auto started = server.start();
    if (!started) { throw std::runtime_error(started.error().message); }
    Json ready = core.dispatch({{"method","hello"}});
    ready["sites"] = Json::array({site});
    // Lossless encoding is the same format used by metallicctl.
    save(directory / "Server.json", debug::encodeLossless(ready));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    uint64_t execution = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        core.expire();
        for (const auto& request : core.takeShaderRequests("shader-trace-fixture", 1)) {
            const auto begun = runtime.begin(request, site, {{"graph",request.graph}, {"generation",request.generation},
                {"execution",++execution}, {"pass","ShaderTraceFixture"}, {"phase","fixture"}, {"dispatchOrdinal",0},
                {"sourceHashes",sources}, {"commandBufferRecording",execution}, {"inputFingerprint",debug::debugSha256(Json{{"value",73},{"cookie",305397763}}.dump())},
                {"variant",{{"module","ShaderTraceFixture"}, {"entry","heapMain"}, {"descriptorMode",mode}, {"instrumentation","Printf"}}}});
            if (!begun) { core.fail(request.id, begun.error()); continue; }
            auto capture = std::make_unique<vk::ShaderPrintf>();
            Json report{{"gpuCompleted",false}};
            const auto path = directory / ("job-" + request.id);
            std::filesystem::create_directory(path);
            try {
                run(report, *capture, path, mode, "none", &runtime, request.specification);
            } catch (const std::exception& e) { report["error"] = e.what(); }
            // run() owns and destroys its device/instance before returning, including error unwinding.
            runtime.finish(*capture, report.value("gpuCompleted",false), true,
                report.contains("readback") && report["readback"].value("passed",false), report);
            save(path / "Job.json", debug::encodeLossless(core.dispatch({{"method","jobs.get"},{"params",{{"job",request.id}}}})));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    server.stop();
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    std::string mode = "ordinary", fault = "none";
    bool service = false;
    uint32_t serveSeconds = 120;
    std::filesystem::path directory;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--serve") { service = true; }
        else if (i + 1 < argc && arg == "--serve-seconds") {
            const std::string value = argv[++i];
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), serveSeconds);
            if (error != std::errc{} || end != value.data() + value.size()) { std::cerr << "Invalid serve duration\n"; return 64; }
        }
        else if (i + 1 < argc && arg == "--mode") { mode = argv[++i]; }
        else if (i + 1 < argc && arg == "--fault") { fault = argv[++i]; }
        else if (i + 1 < argc && arg == "--output") { directory = argv[++i]; }
        else { std::cerr << "Usage: MetallicShaderPrintfProbe --output <new-dir> --mode ordinary|heap-mapped|heap-native [--fault none|no-info|stdout|gpu-overflow] [--serve --serve-seconds 120]\n"; return 64; }
    }
    if ((service && fault != "none") || !serveSeconds || serveSeconds > 300 || directory.empty() || std::filesystem::exists(directory) ||
        (mode != "ordinary" && mode != "heap-mapped" && mode != "heap-native") ||
        (fault != "none" && fault != "no-info" && fault != "stdout" && fault != "gpu-overflow")) {
        std::cerr << "Invalid arguments or output directory already exists\n";
        return 64;
    }
    try {
        std::filesystem::create_directories(directory);
        if (service) { return serve(directory, mode, serveSeconds); }
        auto capture = std::make_unique<vk::ShaderPrintf>(vk::ShaderPrintfOptions{
            .bufferBytes = fault == "gpu-overflow" ? 128u : 65536u,
            .subscribeInfo = fault != "no-info", .toStdout = fault == "stdout"});
        Json report{{"schema", "metallic.shader-printf.p0.v1"}, {"status", "running"}, {"mode", mode}, {"fault", fault},
            {"backend", "VVL.DebugPrintf"}, {"instrumentation", true}, {"performanceEligible", false},
            {"smokeVerified", false}, {"gpuCompleted", false}, {"expectedRecords", 0},
            {"settings", {{"bufferBytes", capture->options().bufferBytes}, {"info", capture->options().subscribeInfo},
                {"stdout", capture->options().toStdout}, {"printfOnly", false}}}};
        save(directory / "Report.json", report);
        try {
            if (!SDL_Init(SDL_INIT_VIDEO)) { throw std::runtime_error(SDL_GetError()); }
            run(report, *capture, directory, mode, fault);
        } catch (const std::exception& error) {
            report["error"] = error.what();
        }
        SDL_Quit();
        report["capabilities"] = {{"layerDiscovered", capture->layerDiscovered},
            {"layerSpecVersion", version(capture->layerSpecVersion)}, {"layerImplementationVersion", capture->layerImplementationVersion},
            {"instanceConfigured", capture->instanceConfigured}, {"messengerConfigured", capture->messengerConfigured},
            {"deviceConfigured", capture->deviceConfigured}};
        Json messages = Json::array();
        std::multiset<std::string> actual;
        bool warningOrError = false;
        bool gpuOverflow = false;
        for (const auto& message : capture->snapshot()) {
            messages.push_back({{"id", message.id}, {"idName", message.idName.data()}, {"severity", message.severity},
                {"text", message.text.data()}, {"truncated", message.truncated}});
            warningOrError |= (message.severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) != 0;
            const std::string text(message.text.data());
            gpuOverflow |= message.id == 0x4fe1fef9 && text.find("[WARNING]") != std::string::npos &&
                text.find("truncated due to the buffer size") != std::string::npos;
            const auto start = text.find("MTP0 ");
            // Require the VVL Printf message ID and INFO severity, not arbitrary matching log text.
            if (message.id == 0x4fe1fef9 && message.severity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT && start != std::string::npos) {
                actual.insert(text.substr(start, text.find_first_of("\r\n", start) - start));
            }
        }
        std::multiset<std::string> expected;
        for (uint32_t i = 0; i < report["expectedRecords"].get<uint32_t>(); ++i) {
            expected.insert(std::string("MTP0 ") + (mode == "ordinary" ? "ordinary" : "heap") + " seq=" + std::to_string(i) +
                " group=1 lane=3 value=73" + (mode == "ordinary" ? "" : " cookie=305397763"));
        }
        save(directory / "RawMessages.json", messages);
        report["receivedRecords"] = actual.size();
        report["echoMatches"] = !expected.empty() && actual == expected;
        report["hostDropped"] = capture->dropped();
        report["hostTruncated"] = capture->truncated();
        report["warningOrError"] = warningOrError;
        report["gpuOverflow"] = gpuOverflow;
        const bool passed = !report.contains("error") && report["gpuCompleted"].get<bool>() && report["echoMatches"].get<bool>() &&
            !warningOrError && !gpuOverflow && capture->dropped() == 0 && capture->truncated() == 0;
        report["smokeVerified"] = passed;
        report["phase"] = "finished";
        report["status"] = passed ? "verified" : report.contains("error") ? "failed" : "incomplete";
        save(directory / "Report.json", report);
        std::cout << report.dump(2) << '\n';
        return passed ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 3;
    }
}

