#include "Runtime/Render/Debug/GpuDebugProbe.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace metallic::render {
using debug::DebugValue;
namespace {
[[noreturn]] void reject(std::string code, std::string message)
{
    throw debug::DebugError{std::move(code), std::move(message)};
}

uint32_t scalarBits(const DebugValue& value, std::string_view type)
{
    if (type == "u32") { return static_cast<uint32_t>(debug::debugUnsigned(value, UINT32_MAX)); }
    if (type == "i32") {
        if (!value.is_number_integer() || (value.is_number_unsigned() ? value.get<uint64_t>() > uint64_t(INT32_MAX) :
            value.get<int64_t>() < INT32_MIN || value.get<int64_t>() > INT32_MAX)) {
            reject("TypeMismatch", "Expected an i32 threshold");
        }
        return std::bit_cast<uint32_t>(value.get<int32_t>());
    }
    if (!value.is_number() || !std::isfinite(value.get<double>()) || !std::isfinite(value.get<float>())) {
        reject("TypeMismatch", "Expected a finite f32 threshold");
    }
    return std::bit_cast<uint32_t>(value.get<float>());
}
} // namespace

debug::DebugResult<std::vector<PreparedDebugProbe>> prepareDebugProbes(
    const DebugValue& specification, std::span<const DebugResourceBinding> resources,
    const std::unordered_map<std::string, debug::DebugTypeDesc>& layouts,
    const debug::DebugEvidenceStamp& evidence, uint64_t scanBudget)
{
    try {
        std::vector<PreparedDebugProbe> probes;
        uint64_t total = 0;
        for (const auto& spec : specification.value("probes", DebugValue::array())) {
            const auto id = spec.at("id").get<std::string>();
            auto source = std::find_if(resources.begin(), resources.end(), [&](const auto& value) { return value.id == id; });
            if (source == resources.end()) { reject("NotFound", "Probe resource unavailable at this checkpoint: " + id); }
            if (!source->buffer || source->state == ResourceState::Undefined || !source->metadata.value("captureSupported", true) ||
                !(uint32_t(source->buffer->desc().usage) & uint32_t(BufferUsageBits::Storage))) {
                reject("Unsupported", "Probe requires a produced storage buffer with an explicit owner state: " + id);
            }
            const auto allocation = source->allocation ? source->allocation : evidence.generation;
            if (spec.contains("allocation") && debug::debugUnsigned(spec.at("allocation")) != allocation) {
                reject("StaleHandle", "Probe allocation changed");
            }
            const auto name = spec.value("layout", source->layout);
            if (!layouts.contains(name) || (source->layout != "raw" && source->layout != name)) {
                reject("LayoutMismatch", "Probe requires the registered source layout");
            }
            const auto& layout = layouts.at(name);
            if (spec.contains("layoutHash") && spec.at("layoutHash") != layout.layoutHash()) { reject("LayoutMismatch", "Probe layout hash changed"); }
            const auto fieldName = spec.value("field", "value");
            auto field = std::find_if(layout.fields.begin(), layout.fields.end(), [&](const auto& f) { return f.name == fieldName; });
            if (field == layout.fields.end()) { reject("LayoutMismatch", "Unknown probe field"); }
            if (field->type != "u32" && field->type != "i32" && field->type != "f32") { reject("Unsupported", "GPU probes support u32/i32/f32 fields"); }
            const auto component = debug::debugUnsigned(spec.value("component", DebugValue(0)));
            if (component >= field->count) { reject("OutOfRange", "Vector component outside field"); }
            if (!layout.stride || layout.stride % 4 || field->offset % 4 || field->offset + component * 4 + 4 > layout.stride ||
                field->bitOffset >= 32 || field->bitWidth > 32 - field->bitOffset || field->scale > UINT32_MAX ||
                (field->bitWidth && field->type != "u32")) { reject("Unsupported", "Unsupported GPU field alignment or bit layout"); }
            const auto mask = field->bitWidth ? UINT32_MAX >> (32 - field->bitWidth) : UINT32_MAX;
            if (uint64_t(mask) * field->scale > UINT32_MAX) { reject("Unsupported", "Scaled GPU field exceeds u32"); }
            const uint64_t offset = debug::debugUnsigned(spec.value("offset", DebugValue(0)));
            const uint64_t count = debug::debugUnsigned(spec.at("count"), 16u << 20);
            const auto& desc = source->buffer->desc();
            if (source->offset > desc.size) { reject("OutOfRange", "Invalid owner view offset"); }
            const auto available = source->size ? source->size : desc.size - source->offset;
            if (available > desc.size - source->offset || offset > available / layout.stride || !count || count > available / layout.stride - offset) {
                reject("OutOfRange", "Probe interval exceeds source view");
            }
            const auto byteOffset = source->offset + offset * layout.stride, scanBytes = count * layout.stride;
            if (byteOffset % 4 || byteOffset > UINT32_MAX || scanBytes > uint64_t(UINT32_MAX) + 1 - byteOffset) {
                reject("Unsupported", "GPU probe address must fit 32-bit byte addressing");
            }
            if (scanBytes > scanBudget - total) { reject("BudgetExceeded", "GPU probe scan budget exceeded"); }
            total += scanBytes;
            const auto operation = spec.at("operation").get<std::string>();
            if (operation == "nonFinite" && field->type != "f32") { reject("TypeMismatch", "nonFinite requires f32"); }
            if (operation == "outOfBounds" && field->type == "f32") { reject("TypeMismatch", "Index bounds require u32 or i32"); }
            const auto lower = operation == "outOfBounds" ? spec.value("lower", DebugValue(0)) : spec.value("value", DebugValue(0));
            const auto upper = spec.value("upper", DebugValue(0));
            const uint32_t lowerBits = scalarBits(lower, field->type), upperBits = scalarBits(upper, field->type);
            if (operation == "outOfBounds" && (field->type == "u32" ? lowerBits >= upperBits :
                std::bit_cast<int32_t>(lowerBits) >= std::bit_cast<int32_t>(upperBits))) {
                reject("InvalidArgument", "Index interval must have lower < upper");
            }
            const std::vector<std::string> predicates{"all", "eq", "ne", "lt", "le", "gt", "ge"};
            const auto predicate = std::find(predicates.begin(), predicates.end(), spec.value("predicate", "all"));
            PreparedDebugProbe probe;
            probe.source = &*source; probe.scanBytes = scanBytes;
            probe.push = {static_cast<uint32_t>(byteOffset), layout.stride, static_cast<uint32_t>(field->offset + component * 4),
                static_cast<uint32_t>(count), field->type == "u32" ? 0u : field->type == "i32" ? 1u : 2u,
                operation == "count" ? 0u : operation == "outOfBounds" ? 1u : operation == "nonFinite" ? 2u : 3u,
                static_cast<uint32_t>(predicate - predicates.begin()), lowerBits, upperBits,
                static_cast<uint32_t>(std::min(uint64_t(256), (count + 127) / 128)), field->bitOffset, mask, static_cast<uint32_t>(field->scale)};
            const auto probeName = spec.value("name", id);
            const DebugValue coverage{{"elementOffset", offset}, {"elementCount", count}, {"capacity", available / layout.stride},
                {"completeCoverage", offset == 0 && scanBytes == available}, {"storageOnly", true}};
            probe.metadata = {{"id", "probe." + probeName}, {"name", probeName}, {"kind", "gpuProbe"},
                {"source", id}, {"operation", operation}, {"field", fieldName}, {"component", component}, {"scalarType", field->type},
                {"sourceLayout", layout.schema()}, {"allocation", allocation}, {"evidence", evidence.value()},
                {"elementOffset", offset}, {"elementCount", count}, {"coverage", coverage}, {"configuration", spec},
                {"scanBytes", scanBytes}, {"readbackBytes", probe.push.groupCount * 32u}, {"thresholdBits", lowerBits},
                {"backend", "fixed-slang-v1-group-reduction"}, {"captured", false}};
            probes.push_back(std::move(probe));
        }
        return probes;
    } catch (const debug::DebugError& error) { return std::unexpected(error); }
    catch (const std::exception& error) { return std::unexpected(debug::DebugError{"InvalidArgument", error.what()}); }
}

Result initializeDebugProbe(Device& device, ComputeProgram& program, std::string& log)
{
    if (program.valid()) { return {}; }
    ShaderCompileResult shader;
    auto result = compileSlangShaderToSpirv({.moduleName = "GpuProbe", .entryPointName = "probe",
        .searchPath = PROJECT_SOURCE_DIR "/Shaders/Debug"}, shader);
    if (!result) { log = shader.diagnostics; return result; }
    const ComputeProgramBindingDesc bindings[] = {{0}, {1}};
    return program.initialize(device, {.spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
        .pushConstantSize = sizeof(DebugProbePush), .bindings = bindings, .bindingCount = 2,
        .debugName = "DebugGpuProbe", .requiresRayQuery = false}, log);
}

Result recordDebugProbe(CommandBuffer& commands, ComputeProgram& program,
    const PreparedDebugProbe& probe, Buffer& output, Buffer& readback)
{
    BufferBarrierDesc source{.buffer = probe.source->buffer, .before = probe.source->state, .after = ResourceState::ShaderRead,
        .offset = probe.push.byteOffset, .size = probe.scanBytes};
    BufferBarrierDesc destination{.buffer = &output, .before = ResourceState::Undefined, .after = ResourceState::General};
    commands.barrier({.buffers = &source, .bufferCount = 1});
    commands.barrier({.buffers = &destination, .bufferCount = 1});
    const ComputeDispatchBinding bindings[] = {{.binding = 0, .buffer = probe.source->buffer}, {.binding = 1, .buffer = &output}};
    const auto result = program.dispatch({.commandBuffer = &commands, .bindings = bindings, .bindingCount = 2,
        .pushData = &probe.push, .pushDataSize = sizeof(probe.push), .groupCountX = probe.push.groupCount});
    std::swap(source.before, source.after);
    commands.barrier({.buffers = &source, .bufferCount = 1});
    if (!result) { return result; }
    destination.before = ResourceState::General; destination.after = ResourceState::TransferSource;
    commands.barrier({.buffers = &destination, .bufferCount = 1});
    commands.copyBuffer({.source = &output, .destination = &readback, .size = readback.desc().size});
    return {};
}

} // namespace metallic::render
