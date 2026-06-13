#include "Runtime/Render/RenderGraph/render_graph.h"

#include "Runtime/Render/slang_compiler.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

constexpr const char* kTriangleShaderSearchPath = PROJECT_SOURCE_DIR "/Shaders";
constexpr const char* kTriangleShaderModuleName = "triangle";
constexpr const char* kTriangleVertexEntryPoint = "triangleVertexMain";
constexpr const char* kTriangleFragmentEntryPoint = "triangleFragmentMain";

struct RenderGraphPassRegistryEntry {
    std::string description;
    RenderGraphPassFactory factory;
};

std::unordered_map<std::string, RenderGraphPassRegistryEntry>& passRegistry()
{
    static std::unordered_map<std::string, RenderGraphPassRegistryEntry> registry;
    return registry;
}

bool isOutputMarked(const RenderGraph& graph, std::string_view fullName)
{
    for (const RenderGraphOutput& output : graph.outputs()) {
        if (makeRenderGraphFieldName(output.passName, output.fieldName) == fullName) {
            return true;
        }
    }
    return false;
}

TextureUsageBits addTextureUsage(TextureUsageBits usage, TextureUsageBits flag)
{
    return usage | flag;
}

Format resolveFormat(Format format, Format defaultFormat)
{
    return format == Format::Unknown ? defaultFormat : format;
}

std::string resultMessage(std::string_view label, const Result& result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

bool nodeNameExists(const std::vector<RenderGraphNode>& nodes, std::string_view name, uint32_t ignoreId = 0)
{
    return std::any_of(
        nodes.begin(),
        nodes.end(),
        [name, ignoreId](const RenderGraphNode& node) {
            return node.id != ignoreId && node.name == name;
        });
}

const RenderGraphNode* findNodeByName(const std::vector<RenderGraphNode>& nodes, std::string_view name)
{
    const auto iter = std::find_if(
        nodes.begin(),
        nodes.end(),
        [name](const RenderGraphNode& node) {
            return node.name == name;
        });
    return iter == nodes.end() ? nullptr : &(*iter);
}

std::string validationPrefix(std::string_view issue)
{
    std::string message("RenderGraph validation failed: ");
    message += issue;
    return message;
}

bool validateAcyclic(
    const std::vector<RenderGraphNode>& nodes,
    const std::vector<RenderGraphEdge>& edges,
    std::string& log)
{
    std::unordered_map<std::string, uint32_t> indegree;
    std::unordered_map<std::string, std::vector<std::string>> outgoing;

    for (const RenderGraphNode& node : nodes) {
        indegree.emplace(node.name, 0);
    }

    for (const RenderGraphEdge& edge : edges) {
        if (indegree.find(edge.srcPass) == indegree.end() || indegree.find(edge.dstPass) == indegree.end()) {
            continue;
        }
        outgoing[edge.srcPass].push_back(edge.dstPass);
        ++indegree[edge.dstPass];
    }

    std::queue<std::string> ready;
    for (const auto& [name, degree] : indegree) {
        if (degree == 0) {
            ready.push(name);
        }
    }

    size_t visited = 0;
    while (!ready.empty()) {
        std::string current = ready.front();
        ready.pop();
        ++visited;

        for (const std::string& next : outgoing[current]) {
            auto iter = indegree.find(next);
            if (iter == indegree.end()) {
                continue;
            }
            if (--iter->second == 0) {
                ready.push(next);
            }
        }
    }

    if (visited != nodes.size()) {
        log = validationPrefix("cycle detected");
        return false;
    }
    return true;
}

class ClearColorPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addOutput("color", "Cleared color target")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        RenderGraphResource* color = context.output("color");
        if (color == nullptr || color->view == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        ColorValue clear{0.04f, 0.06f, 0.09f, 1.0f};
        const RenderGraphProperties& props = context.properties();
        if (props.contains("color") && props["color"].is_array() && props["color"].size() >= 4) {
            clear.r = props["color"][0].get<float>();
            clear.g = props["color"][1].get<float>();
            clear.b = props["color"][2].get<float>();
            clear.a = props["color"][3].get<float>();
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color->view,
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = clear,
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().endRendering();
        return {};
    }
};

class CopyColorPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        RenderGraphField& source = reflection.addInput("source", "Source color texture");
        source.usage = TextureUsageBits::TransferSource;
        source.state = ResourceState::TransferSource;

        RenderGraphField& color = reflection.addOutput("color", "Copied color texture");
        color.usage = TextureUsageBits::TransferDestination;
        color.state = ResourceState::TransferDestination;
        color.format = Format::Rgba8Unorm;
        return reflection;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        RenderGraphResource* source = context.input("source");
        RenderGraphResource* color = context.output("color");
        if (source == nullptr ||
            source->texture == nullptr ||
            color == nullptr ||
            color->texture == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        context.commandBuffer().copyTexture(TextureCopyDesc{
            .source = source->texture,
            .destination = color->texture,
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .sourceMipLevel = 0,
            .sourceBaseLayer = 0,
            .destinationMipLevel = 0,
            .destinationBaseLayer = 0,
        });
        return {};
    }
};

class TriangleRasterPass final : public RenderGraphPass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addOutput("color", "Rasterized triangle color")
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result result = createShaderModule(*context.device, kTriangleVertexEntryPoint, vertexShader_, log);
        if (!result) {
            return result;
        }
        result = createShaderModule(*context.device, kTriangleFragmentEntryPoint, fragmentShader_, log);
        if (!result) {
            return result;
        }

        result = context.device->createGraphicsPipeline(
            GraphicsPipelineDesc{
                .vertexShader = vertexShader_.get(),
                .fragmentShader = fragmentShader_.get(),
                .colorFormat = Format::Rgba8Unorm,
                .topology = PrimitiveTopology::TriangleList,
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createGraphicsPipeline", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        RenderGraphResource* color = context.output("color");
        if (color == nullptr || color->view == nullptr || pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const Rect renderArea{
            .x = 0,
            .y = 0,
            .width = context.width(),
            .height = context.height(),
        };
        RenderingAttachmentDesc attachment{
            .view = color->view,
            .state = ResourceState::ColorAttachment,
            .loadOp = LoadOp::Clear,
            .storeOp = StoreOp::Store,
            .clearColor = ColorValue{0.04f, 0.06f, 0.09f, 1.0f},
        };
        context.commandBuffer().beginRendering(RenderingDesc{
            .renderArea = renderArea,
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().setViewport(Viewport{
            .x = 0.0f,
            .y = 0.0f,
            .width = static_cast<float>(context.width()),
            .height = static_cast<float>(context.height()),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        });
        context.commandBuffer().setScissor(renderArea);
        context.commandBuffer().bindGraphicsPipeline(*pipeline_);
        context.commandBuffer().draw(3);
        context.commandBuffer().endRendering();
        return {};
    }

private:
    static Result createShaderModule(
        Device& device,
        const char* entryPointName,
        std::unique_ptr<ShaderModule>& outShaderModule,
        std::string& log)
    {
        ShaderCompileResult compileResult;
        Result result = compileSlangShaderToSpirv(
            SlangShaderDesc{
                .moduleName = kTriangleShaderModuleName,
                .entryPointName = entryPointName,
                .searchPath = kTriangleShaderSearchPath,
            },
            compileResult);
        if (!result) {
            log += "compileSlangShaderToSpirv(";
            log += entryPointName;
            log += ") returned ";
            log += resultToString(result);
            if (!compileResult.diagnostics.empty()) {
                log += ": ";
                log += compileResult.diagnostics;
            }
            log += '\n';
            return result;
        }

        result = device.createShaderModule(
            ShaderModuleDesc{
                .code = compileResult.spirv.data(),
                .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            },
            outShaderModule);
        if (!result) {
            log += resultMessage("createShaderModule", result);
            log += '\n';
        }
        return result;
    }

    std::unique_ptr<ShaderModule> vertexShader_;
    std::unique_ptr<ShaderModule> fragmentShader_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
};

struct ActiveGraph {
    std::unordered_set<std::string> activePasses;
    std::vector<std::string> executionOrder;
};

bool buildActiveGraph(const RenderGraph& graph, ActiveGraph& activeGraph, std::string& log)
{
    std::unordered_map<std::string, std::vector<std::string>> incoming;
    for (const RenderGraphEdge& edge : graph.edges()) {
        incoming[edge.dstPass].push_back(edge.srcPass);
    }

    std::function<void(const std::string&)> visitInputs = [&](const std::string& passName) {
        if (!activeGraph.activePasses.insert(passName).second) {
            return;
        }
        for (const std::string& srcPass : incoming[passName]) {
            visitInputs(srcPass);
        }
    };

    for (const RenderGraphOutput& output : graph.outputs()) {
        visitInputs(output.passName);
    }

    std::unordered_map<std::string, uint32_t> indegree;
    std::unordered_map<std::string, std::vector<std::string>> outgoing;
    for (const std::string& passName : activeGraph.activePasses) {
        indegree.emplace(passName, 0);
    }
    for (const RenderGraphEdge& edge : graph.edges()) {
        if (!activeGraph.activePasses.contains(edge.srcPass) ||
            !activeGraph.activePasses.contains(edge.dstPass)) {
            continue;
        }
        outgoing[edge.srcPass].push_back(edge.dstPass);
        ++indegree[edge.dstPass];
    }

    std::queue<std::string> ready;
    for (const auto& [name, degree] : indegree) {
        if (degree == 0) {
            ready.push(name);
        }
    }

    while (!ready.empty()) {
        std::string current = ready.front();
        ready.pop();
        activeGraph.executionOrder.push_back(current);

        for (const std::string& next : outgoing[current]) {
            auto iter = indegree.find(next);
            if (iter == indegree.end()) {
                continue;
            }
            if (--iter->second == 0) {
                ready.push(next);
            }
        }
    }

    if (activeGraph.executionOrder.size() != activeGraph.activePasses.size()) {
        log = validationPrefix("cycle detected in active graph");
        return false;
    }
    return true;
}

} // namespace

RenderGraphField& RenderPassReflection::addInput(std::string name, std::string description)
{
    fields_.push_back(RenderGraphField{
        .name = std::move(name),
        .description = std::move(description),
        .visibility = RenderGraphFieldVisibility::Input,
        .format = Format::Rgba8Unorm,
        .usage = TextureUsageBits::Sampled,
        .state = ResourceState::ShaderRead,
    });
    return fields_.back();
}

RenderGraphField& RenderPassReflection::addOutput(std::string name, std::string description)
{
    fields_.push_back(RenderGraphField{
        .name = std::move(name),
        .description = std::move(description),
        .visibility = RenderGraphFieldVisibility::Output,
        .format = Format::Rgba8Unorm,
        .usage = TextureUsageBits::ColorAttachment,
        .state = ResourceState::ColorAttachment,
    });
    return fields_.back();
}

const RenderGraphField* RenderPassReflection::findField(
    std::string_view name,
    RenderGraphFieldVisibility visibility) const
{
    const auto iter = std::find_if(
        fields_.begin(),
        fields_.end(),
        [name, visibility](const RenderGraphField& field) {
            return field.visibility == visibility && field.name == name;
        });
    return iter == fields_.end() ? nullptr : &(*iter);
}

Result RenderGraphPass::compile(const RenderGraphCompileContext&, std::string&)
{
    return {};
}

RenderGraphExecutionContext::RenderGraphExecutionContext(
    CommandBuffer& commandBuffer,
    uint32_t width,
    uint32_t height,
    const RenderGraphProperties& properties,
    std::vector<Binding> bindings)
    : commandBuffer_(commandBuffer)
    , width_(width)
    , height_(height)
    , properties_(properties)
    , bindings_(std::move(bindings))
{
}

RenderGraphResource* RenderGraphExecutionContext::resource(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.fieldName == fieldName;
        });
    return iter == bindings_.end() ? nullptr : iter->resource;
}

RenderGraphResource* RenderGraphExecutionContext::input(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.visibility == RenderGraphFieldVisibility::Input &&
                binding.fieldName == fieldName;
        });
    return iter == bindings_.end() ? nullptr : iter->resource;
}

RenderGraphResource* RenderGraphExecutionContext::output(std::string_view fieldName) const
{
    const auto iter = std::find_if(
        bindings_.begin(),
        bindings_.end(),
        [fieldName](const Binding& binding) {
            return binding.visibility == RenderGraphFieldVisibility::Output &&
                binding.fieldName == fieldName;
        });
    return iter == bindings_.end() ? nullptr : iter->resource;
}

bool registerRenderGraphPassType(
    std::string type,
    std::string description,
    RenderGraphPassFactory factory)
{
    if (type.empty() || !factory) {
        return false;
    }
    passRegistry()[std::move(type)] = RenderGraphPassRegistryEntry{
        .description = std::move(description),
        .factory = std::move(factory),
    };
    return true;
}

void registerBuiltInRenderGraphPasses()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;

    registerRenderGraphPassType(
        "ClearColorPass",
        "Clear a color texture",
        []() { return std::make_unique<ClearColorPass>(); });
    registerRenderGraphPassType(
        "CopyColorPass",
        "Copy a color texture",
        []() { return std::make_unique<CopyColorPass>(); });
    registerRenderGraphPassType(
        "TriangleRasterPass",
        "Rasterize the built-in triangle shader",
        []() { return std::make_unique<TriangleRasterPass>(); });
}

std::unique_ptr<RenderGraphPass> createRenderGraphPass(std::string_view type)
{
    registerBuiltInRenderGraphPasses();
    const auto iter = passRegistry().find(std::string(type));
    if (iter == passRegistry().end() || !iter->second.factory) {
        return {};
    }
    return iter->second.factory();
}

std::vector<RenderGraphPassInfo> listRenderGraphPassTypes()
{
    registerBuiltInRenderGraphPasses();
    std::vector<RenderGraphPassInfo> passTypes;
    passTypes.reserve(passRegistry().size());
    for (const auto& [type, entry] : passRegistry()) {
        passTypes.push_back(RenderGraphPassInfo{
            .type = type,
            .description = entry.description,
        });
    }
    std::sort(
        passTypes.begin(),
        passTypes.end(),
        [](const RenderGraphPassInfo& lhs, const RenderGraphPassInfo& rhs) {
            return lhs.type < rhs.type;
        });
    return passTypes;
}

RenderGraph::RenderGraph()
{
    registerBuiltInRenderGraphPasses();
}

void RenderGraph::setName(std::string name)
{
    if (name.empty()) {
        name = "RenderGraph";
    }
    if (name_ != name) {
        name_ = std::move(name);
        markDirty();
    }
}

const RenderGraphNode* RenderGraph::findNode(std::string_view name) const
{
    return findNodeByName(nodes_, name);
}

RenderGraphNode* RenderGraph::findNode(std::string_view name)
{
    return const_cast<RenderGraphNode*>(static_cast<const RenderGraph*>(this)->findNode(name));
}

const RenderGraphNode* RenderGraph::findNode(uint32_t id) const
{
    const auto iter = std::find_if(
        nodes_.begin(),
        nodes_.end(),
        [id](const RenderGraphNode& node) {
            return node.id == id;
        });
    return iter == nodes_.end() ? nullptr : &(*iter);
}

RenderGraphNode* RenderGraph::findNode(uint32_t id)
{
    return const_cast<RenderGraphNode*>(static_cast<const RenderGraph*>(this)->findNode(id));
}

const RenderGraphEdge* RenderGraph::findEdge(uint32_t id) const
{
    const auto iter = std::find_if(
        edges_.begin(),
        edges_.end(),
        [id](const RenderGraphEdge& edge) {
            return edge.id == id;
        });
    return iter == edges_.end() ? nullptr : &(*iter);
}

RenderGraphNode* RenderGraph::addNode(
    std::string type,
    std::string name,
    RenderGraphProperties properties,
    float uiX,
    float uiY)
{
    if (type.empty() || name.empty() || nodeNameExists(nodes_, name)) {
        return nullptr;
    }
    nodes_.push_back(RenderGraphNode{
        .id = nextNodeId_++,
        .name = std::move(name),
        .type = std::move(type),
        .properties = std::move(properties),
        .uiX = uiX,
        .uiY = uiY,
    });
    markDirty();
    return &nodes_.back();
}

bool RenderGraph::removeNode(uint32_t id)
{
    const RenderGraphNode* node = findNode(id);
    if (node == nullptr) {
        return false;
    }
    const std::string nodeName = node->name;
    nodes_.erase(
        std::remove_if(
            nodes_.begin(),
            nodes_.end(),
            [id](const RenderGraphNode& candidate) { return candidate.id == id; }),
        nodes_.end());
    edges_.erase(
        std::remove_if(
            edges_.begin(),
            edges_.end(),
            [&nodeName](const RenderGraphEdge& edge) {
                return edge.srcPass == nodeName || edge.dstPass == nodeName;
            }),
        edges_.end());
    outputs_.erase(
        std::remove_if(
            outputs_.begin(),
            outputs_.end(),
            [&nodeName](const RenderGraphOutput& output) {
                return output.passName == nodeName;
            }),
        outputs_.end());
    markDirty();
    return true;
}

bool RenderGraph::renameNode(uint32_t id, std::string newName)
{
    if (newName.empty() || nodeNameExists(nodes_, newName, id)) {
        return false;
    }
    RenderGraphNode* node = findNode(id);
    if (node == nullptr || node->name == newName) {
        return node != nullptr;
    }
    const std::string oldName = node->name;
    node->name = std::move(newName);
    for (RenderGraphEdge& edge : edges_) {
        if (edge.srcPass == oldName) {
            edge.srcPass = node->name;
        }
        if (edge.dstPass == oldName) {
            edge.dstPass = node->name;
        }
    }
    for (RenderGraphOutput& output : outputs_) {
        if (output.passName == oldName) {
            output.passName = node->name;
        }
    }
    markDirty();
    return true;
}

bool RenderGraph::setNodeProperties(uint32_t id, RenderGraphProperties properties)
{
    RenderGraphNode* node = findNode(id);
    if (node == nullptr) {
        return false;
    }
    node->properties = std::move(properties);
    markDirty();
    return true;
}

bool RenderGraph::setNodePosition(uint32_t id, float uiX, float uiY)
{
    RenderGraphNode* node = findNode(id);
    if (node == nullptr) {
        return false;
    }
    if (node->uiX == uiX && node->uiY == uiY) {
        return true;
    }
    node->uiX = uiX;
    node->uiY = uiY;
    return true;
}

RenderGraphEdge* RenderGraph::addEdge(std::string src, std::string dst)
{
    std::string srcPass;
    std::string srcField;
    std::string dstPass;
    std::string dstField;
    if (!splitRenderGraphFieldName(src, srcPass, srcField) ||
        !splitRenderGraphFieldName(dst, dstPass, dstField)) {
        return nullptr;
    }

    const auto exists = std::any_of(
        edges_.begin(),
        edges_.end(),
        [&](const RenderGraphEdge& edge) {
            return edge.srcPass == srcPass &&
                edge.srcField == srcField &&
                edge.dstPass == dstPass &&
                edge.dstField == dstField;
        });
    if (exists) {
        return nullptr;
    }

    edges_.push_back(RenderGraphEdge{
        .id = nextEdgeId_++,
        .srcPass = std::move(srcPass),
        .srcField = std::move(srcField),
        .dstPass = std::move(dstPass),
        .dstField = std::move(dstField),
    });
    markDirty();
    return &edges_.back();
}

bool RenderGraph::removeEdge(uint32_t id)
{
    const auto oldSize = edges_.size();
    edges_.erase(
        std::remove_if(
            edges_.begin(),
            edges_.end(),
            [id](const RenderGraphEdge& edge) { return edge.id == id; }),
        edges_.end());
    if (edges_.size() == oldSize) {
        return false;
    }
    markDirty();
    return true;
}

bool RenderGraph::markOutput(std::string output)
{
    std::string passName;
    std::string fieldName;
    if (!splitRenderGraphFieldName(output, passName, fieldName)) {
        return false;
    }
    const auto exists = std::any_of(
        outputs_.begin(),
        outputs_.end(),
        [&](const RenderGraphOutput& candidate) {
            return candidate.passName == passName && candidate.fieldName == fieldName;
        });
    if (exists) {
        return true;
    }
    outputs_.push_back(RenderGraphOutput{
        .passName = std::move(passName),
        .fieldName = std::move(fieldName),
    });
    markDirty();
    return true;
}

bool RenderGraph::unmarkOutput(std::string output)
{
    std::string passName;
    std::string fieldName;
    if (!splitRenderGraphFieldName(output, passName, fieldName)) {
        return false;
    }
    const auto oldSize = outputs_.size();
    outputs_.erase(
        std::remove_if(
            outputs_.begin(),
            outputs_.end(),
            [&](const RenderGraphOutput& candidate) {
                return candidate.passName == passName && candidate.fieldName == fieldName;
            }),
        outputs_.end());
    if (outputs_.size() == oldSize) {
        return false;
    }
    markDirty();
    return true;
}

void RenderGraph::clearOutputs()
{
    if (!outputs_.empty()) {
        outputs_.clear();
        markDirty();
    }
}

bool RenderGraph::validate(std::string& log) const
{
    registerBuiltInRenderGraphPasses();
    log.clear();

    if (nodes_.empty()) {
        log = validationPrefix("graph has no nodes");
        return false;
    }
    if (outputs_.empty()) {
        log = validationPrefix("graph has no marked output");
        return false;
    }

    std::unordered_set<uint32_t> ids;
    std::unordered_set<std::string> names;
    std::unordered_map<std::string, RenderPassReflection> reflections;
    const RenderGraphCompileContext reflectContext{};

    for (const RenderGraphNode& node : nodes_) {
        if (node.id == 0 || !ids.insert(node.id).second) {
            log = validationPrefix("duplicate node id");
            return false;
        }
        if (node.name.empty() || !names.insert(node.name).second) {
            log = validationPrefix("duplicate or empty node name");
            return false;
        }
        std::unique_ptr<RenderGraphPass> pass = createRenderGraphPass(node.type);
        if (pass == nullptr) {
            log = validationPrefix(std::string("unknown pass type '") + node.type + "'");
            return false;
        }
        pass->setProperties(node.properties);
        reflections.emplace(node.name, pass->reflect(reflectContext));
    }

    for (const RenderGraphOutput& output : outputs_) {
        const auto iter = reflections.find(output.passName);
        if (iter == reflections.end() ||
            iter->second.findField(output.fieldName, RenderGraphFieldVisibility::Output) == nullptr) {
            log = validationPrefix(
                std::string("invalid output '") +
                makeRenderGraphFieldName(output.passName, output.fieldName) +
                "'");
            return false;
        }
    }

    for (const RenderGraphEdge& edge : edges_) {
        const auto src = reflections.find(edge.srcPass);
        const auto dst = reflections.find(edge.dstPass);
        if (src == reflections.end() ||
            src->second.findField(edge.srcField, RenderGraphFieldVisibility::Output) == nullptr) {
            log = validationPrefix(
                std::string("invalid edge source '") +
                makeRenderGraphFieldName(edge.srcPass, edge.srcField) +
                "'");
            return false;
        }
        if (dst == reflections.end() ||
            dst->second.findField(edge.dstField, RenderGraphFieldVisibility::Input) == nullptr) {
            log = validationPrefix(
                std::string("invalid edge destination '") +
                makeRenderGraphFieldName(edge.dstPass, edge.dstField) +
                "'");
            return false;
        }
    }

    for (const auto& [passName, reflection] : reflections) {
        for (const RenderGraphField& field : reflection.fields()) {
            if (field.visibility != RenderGraphFieldVisibility::Input || field.optional) {
                continue;
            }
            const bool connected = std::any_of(
                edges_.begin(),
                edges_.end(),
                [&](const RenderGraphEdge& edge) {
                    return edge.dstPass == passName && edge.dstField == field.name;
                });
            if (!connected) {
                log = validationPrefix(
                    std::string("required input is not connected '") +
                    makeRenderGraphFieldName(passName, field.name) +
                    "'");
                return false;
            }
        }
    }

    if (!validateAcyclic(nodes_, edges_, log)) {
        return false;
    }

    log = "RenderGraph is valid";
    return true;
}

void RenderGraph::clear()
{
    name_ = "RenderGraph";
    nodes_.clear();
    edges_.clear();
    outputs_.clear();
    nextNodeId_ = 1;
    nextEdgeId_ = 1;
    markDirty();
}

std::string RenderGraph::firstOutputName() const
{
    if (outputs_.empty()) {
        return {};
    }
    return makeRenderGraphFieldName(outputs_.front().passName, outputs_.front().fieldName);
}

RenderGraph RenderGraph::createDefaultTriangleGraph()
{
    RenderGraph graph;
    graph.setName("DefaultTriangle");
    graph.addNode("TriangleRasterPass", "Triangle", RenderGraphProperties::object(), 40.0f, 80.0f);
    graph.markOutput("Triangle.color");
    graph.clearDirty();
    return graph;
}

bool splitRenderGraphFieldName(
    std::string_view fullName,
    std::string& outPassName,
    std::string& outFieldName)
{
    const size_t separator = fullName.find('.');
    if (separator == std::string_view::npos ||
        separator == 0 ||
        separator + 1 >= fullName.size()) {
        return false;
    }
    outPassName = std::string(fullName.substr(0, separator));
    outFieldName = std::string(fullName.substr(separator + 1));
    return true;
}

std::string makeRenderGraphFieldName(std::string_view passName, std::string_view fieldName)
{
    std::string fullName(passName);
    fullName += '.';
    fullName += fieldName;
    return fullName;
}

struct RenderGraphExecutor::Impl {
    struct ResourceSlot {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
        RenderGraphResource resource;
    };

    struct CompiledNode {
        uint32_t id = 0;
        std::string name;
        std::string type;
        RenderGraphProperties properties = RenderGraphProperties::object();
        std::unique_ptr<RenderGraphPass> pass;
        RenderPassReflection reflection;
    };

    Device* device = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    Format defaultFormat = Format::Rgba8Unorm;
    std::vector<CompiledNode> executionList;
    std::unordered_map<std::string, ResourceSlot> resources;
    std::unordered_map<std::string, std::string> inputAliases;
    bool isCompiled = false;

    RenderGraphResource* resource(std::string_view fullName)
    {
        auto iter = resources.find(std::string(fullName));
        return iter == resources.end() ? nullptr : &iter->second.resource;
    }

    const RenderGraphResource* resource(std::string_view fullName) const
    {
        auto iter = resources.find(std::string(fullName));
        return iter == resources.end() ? nullptr : &iter->second.resource;
    }

    const CompiledNode* compiledNode(std::string_view name) const
    {
        const auto iter = std::find_if(
            executionList.begin(),
            executionList.end(),
            [name](const CompiledNode& node) {
                return node.name == name;
            });
        return iter == executionList.end() ? nullptr : &(*iter);
    }

    const RenderGraphField* reflectedField(
        std::string_view passName,
        std::string_view fieldName,
        RenderGraphFieldVisibility visibility) const
    {
        const CompiledNode* node = compiledNode(passName);
        if (node == nullptr) {
            return nullptr;
        }
        return node->reflection.findField(fieldName, visibility);
    }

    Result transition(
        CommandBuffer& commandBuffer,
        RenderGraphResource& resource,
        ResourceState state)
    {
        if (resource.texture == nullptr || resource.state == state) {
            return {};
        }
        TextureBarrierDesc barrier{
            .texture = resource.texture,
            .before = resource.state,
            .after = state,
            .baseMip = 0,
            .mipCount = resource.desc.mipCount,
            .baseLayer = 0,
            .layerCount = resource.desc.layerCount,
        };
        commandBuffer.barrier(BarrierDesc{
            .textures = &barrier,
            .textureCount = 1,
        });
        resource.state = state;
        return {};
    }
};

RenderGraphExecutor::RenderGraphExecutor()
    : impl_(std::make_unique<Impl>())
{
}

RenderGraphExecutor::~RenderGraphExecutor() = default;
RenderGraphExecutor::RenderGraphExecutor(RenderGraphExecutor&&) noexcept = default;
RenderGraphExecutor& RenderGraphExecutor::operator=(RenderGraphExecutor&&) noexcept = default;

Result RenderGraphExecutor::compile(
    Device& device,
    const RenderGraph& graph,
    uint32_t width,
    uint32_t height,
    std::string& log)
{
    if (width == 0 || height == 0) {
        log = validationPrefix("invalid default dimensions");
        return makeError(Error::InvalidArgument);
    }

    std::string validationLog;
    if (!graph.validate(validationLog)) {
        log = validationLog;
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }

    ActiveGraph activeGraph;
    if (!buildActiveGraph(graph, activeGraph, log)) {
        impl_->isCompiled = false;
        return makeError(Error::InvalidArgument);
    }

    impl_->device = &device;
    impl_->width = width;
    impl_->height = height;
    impl_->executionList.clear();
    impl_->resources.clear();
    impl_->inputAliases.clear();
    impl_->isCompiled = false;

    const RenderGraphCompileContext compileContext{
        .device = &device,
        .width = width,
        .height = height,
        .defaultFormat = impl_->defaultFormat,
    };

    for (const std::string& passName : activeGraph.executionOrder) {
        const RenderGraphNode* node = graph.findNode(passName);
        if (node == nullptr) {
            log = validationPrefix(std::string("active pass is missing '") + passName + "'");
            return makeError(Error::InvalidArgument);
        }

        std::unique_ptr<RenderGraphPass> pass = createRenderGraphPass(node->type);
        if (pass == nullptr) {
            log = validationPrefix(std::string("unknown pass type '") + node->type + "'");
            return makeError(Error::InvalidArgument);
        }
        pass->setProperties(node->properties);
        RenderPassReflection reflection = pass->reflect(compileContext);
        Result result = pass->compile(compileContext, log);
        if (!result) {
            impl_->isCompiled = false;
            return result;
        }

        impl_->executionList.push_back(Impl::CompiledNode{
            .id = node->id,
            .name = node->name,
            .type = node->type,
            .properties = node->properties,
            .pass = std::move(pass),
            .reflection = std::move(reflection),
        });
    }

    for (const RenderGraphEdge& edge : graph.edges()) {
        if (!activeGraph.activePasses.contains(edge.srcPass) ||
            !activeGraph.activePasses.contains(edge.dstPass)) {
            continue;
        }
        impl_->inputAliases.emplace(
            makeRenderGraphFieldName(edge.dstPass, edge.dstField),
            makeRenderGraphFieldName(edge.srcPass, edge.srcField));
    }

    for (const Impl::CompiledNode& node : impl_->executionList) {
        for (const RenderGraphField& field : node.reflection.fields()) {
            if (field.visibility != RenderGraphFieldVisibility::Output) {
                continue;
            }

            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            TextureUsageBits usage = field.usage;
            if (usage == TextureUsageBits::None) {
                usage = TextureUsageBits::ColorAttachment;
            }
            if (isOutputMarked(graph, fullName)) {
                usage = addTextureUsage(usage, TextureUsageBits::TransferSource);
            }
            for (const RenderGraphEdge& edge : graph.edges()) {
                if (edge.srcPass != node.name ||
                    edge.srcField != field.name ||
                    !activeGraph.activePasses.contains(edge.dstPass)) {
                    continue;
                }

                const RenderGraphField* dstField = impl_->reflectedField(
                    edge.dstPass,
                    edge.dstField,
                    RenderGraphFieldVisibility::Input);
                usage = addTextureUsage(
                    usage,
                    dstField == nullptr ? TextureUsageBits::Sampled : dstField->usage);
            }

            TextureDesc desc{
                .type = TextureType::Texture2D,
                .usage = usage,
                .format = resolveFormat(field.format, impl_->defaultFormat),
                .width = field.width == 0 ? width : field.width,
                .height = field.height == 0 ? height : field.height,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
            };

            Impl::ResourceSlot slot;
            Result result = device.createTexture(desc, slot.texture);
            if (!result || slot.texture == nullptr) {
                log += resultMessage(std::string("createTexture(") + fullName + ")", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
            result = device.createTextureView(
                *slot.texture,
                TextureViewDesc{
                    .format = desc.format,
                    .baseMip = 0,
                    .mipCount = 1,
                    .baseLayer = 0,
                    .layerCount = 1,
                },
                slot.view);
            if (!result || slot.view == nullptr) {
                log += resultMessage(std::string("createTextureView(") + fullName + ")", result);
                log += '\n';
                return result ? makeError(Error::Failure) : result;
            }
            slot.resource = RenderGraphResource{
                .texture = slot.texture.get(),
                .view = slot.view.get(),
                .desc = desc,
                .state = ResourceState::Undefined,
            };
            impl_->resources.emplace(fullName, std::move(slot));
        }
    }

    impl_->isCompiled = true;
    log = "RenderGraph compiled";
    return {};
}

Result RenderGraphExecutor::execute(CommandBuffer& commandBuffer)
{
    if (!impl_->isCompiled) {
        return makeError(Error::InvalidArgument);
    }

    for (Impl::CompiledNode& node : impl_->executionList) {
        std::vector<RenderGraphExecutionContext::Binding> bindings;

        for (const RenderGraphField& field : node.reflection.fields()) {
            const std::string localName = field.name;
            const std::string fullName = makeRenderGraphFieldName(node.name, field.name);
            RenderGraphResource* resource = nullptr;

            if (field.visibility == RenderGraphFieldVisibility::Output) {
                resource = impl_->resource(fullName);
                if (resource != nullptr) {
                    Result result = impl_->transition(commandBuffer, *resource, field.state);
                    if (!result) {
                        return result;
                    }
                }
            } else {
                const auto alias = impl_->inputAliases.find(fullName);
                if (alias != impl_->inputAliases.end()) {
                    resource = impl_->resource(alias->second);
                    if (resource != nullptr) {
                        Result result = impl_->transition(commandBuffer, *resource, field.state);
                        if (!result) {
                            return result;
                        }
                    }
                }
            }

            bindings.push_back(RenderGraphExecutionContext::Binding{
                .fieldName = localName,
                .resource = resource,
                .visibility = field.visibility,
            });
        }

        RenderGraphExecutionContext context(
            commandBuffer,
            impl_->width,
            impl_->height,
            node.properties,
            std::move(bindings));
        Result result = node.pass->execute(context);
        if (!result) {
            return result;
        }
    }

    return {};
}

Result RenderGraphExecutor::transitionOutput(
    CommandBuffer& commandBuffer,
    std::string_view fullName,
    ResourceState state)
{
    RenderGraphResource* resource = outputResource(fullName);
    if (resource == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    return impl_->transition(commandBuffer, *resource, state);
}

RenderGraphResource* RenderGraphExecutor::outputResource(std::string_view fullName)
{
    return impl_->resource(fullName);
}

const RenderGraphResource* RenderGraphExecutor::outputResource(std::string_view fullName) const
{
    return impl_->resource(fullName);
}

bool RenderGraphExecutor::compiled() const
{
    return impl_->isCompiled;
}

uint32_t RenderGraphExecutor::width() const
{
    return impl_->width;
}

uint32_t RenderGraphExecutor::height() const
{
    return impl_->height;
}

struct RenderGraphPreviewRenderer::Impl {
    std::unique_ptr<Device> device;
    Queue* graphicsQueue = nullptr;
    std::unique_ptr<CommandPool> commandPool;
    std::unique_ptr<CommandBuffer> commandBuffer;
    std::unique_ptr<Fence> fence;
    std::unique_ptr<Buffer> readbackBuffer;
    RenderGraphExecutor executor;
    std::vector<uint32_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t readbackWidth = 0;
    uint32_t readbackHeight = 0;
    std::string lastLog;

    Result ensureReadback(uint32_t newWidth, uint32_t newHeight)
    {
        if (device == nullptr || newWidth == 0 || newHeight == 0) {
            return makeError(Error::InvalidArgument);
        }
        if (readbackBuffer != nullptr && readbackWidth == newWidth && readbackHeight == newHeight) {
            return {};
        }
        readbackBuffer.reset();
        const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
        Result result = device->createBuffer(
            BufferDesc{
                .size = byteSize,
                .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result) {
            return result;
        }
        readbackWidth = newWidth;
        readbackHeight = newHeight;
        pixels.resize(static_cast<size_t>(newWidth) * static_cast<size_t>(newHeight));
        return {};
    }
};

RenderGraphPreviewRenderer::RenderGraphPreviewRenderer()
    : impl_(std::make_unique<Impl>())
{
}

RenderGraphPreviewRenderer::~RenderGraphPreviewRenderer() = default;
RenderGraphPreviewRenderer::RenderGraphPreviewRenderer(RenderGraphPreviewRenderer&&) noexcept = default;
RenderGraphPreviewRenderer& RenderGraphPreviewRenderer::operator=(RenderGraphPreviewRenderer&&) noexcept = default;

Result RenderGraphPreviewRenderer::initialize(bool enableValidation)
{
    Result result = createDevice(
        DeviceDesc{
            .applicationName = "Metallic RenderGraph Preview",
            .enableValidation = enableValidation,
        },
        impl_->device);
    if (!result) {
        return result;
    }

    impl_->graphicsQueue = impl_->device->getQueue(QueueType::Graphics);
    if (impl_->graphicsQueue == nullptr) {
        return makeError(Error::Unsupported);
    }

    result = impl_->device->createCommandPool(*impl_->graphicsQueue, impl_->commandPool);
    if (!result) {
        return result;
    }
    result = impl_->commandPool->createCommandBuffer(impl_->commandBuffer);
    if (!result) {
        return result;
    }
    return impl_->device->createFence(true, impl_->fence);
}

Result RenderGraphPreviewRenderer::render(RenderGraph& graph, uint32_t newWidth, uint32_t newHeight)
{
    if (impl_->device == nullptr ||
        impl_->graphicsQueue == nullptr ||
        impl_->commandPool == nullptr ||
        impl_->commandBuffer == nullptr ||
        impl_->fence == nullptr ||
        newWidth == 0 ||
        newHeight == 0) {
        return makeError(Error::InvalidArgument);
    }

    Result result = impl_->fence->wait();
    if (!result) {
        return result;
    }

    const bool needsCompile =
        graph.dirty() ||
        !impl_->executor.compiled() ||
        impl_->executor.width() != newWidth ||
        impl_->executor.height() != newHeight;
    if (needsCompile) {
        result = impl_->device->waitIdle();
        if (!result) {
            return result;
        }
        result = impl_->executor.compile(
            *impl_->device,
            graph,
            newWidth,
            newHeight,
            impl_->lastLog);
        if (!result) {
            return result;
        }
        graph.clearDirty();
    }

    result = impl_->ensureReadback(newWidth, newHeight);
    if (!result) {
        return result;
    }

    result = impl_->fence->reset();
    if (!result) {
        return result;
    }
    result = impl_->commandPool->reset();
    if (!result) {
        return result;
    }
    result = impl_->commandBuffer->begin();
    if (!result) {
        return result;
    }

    result = impl_->executor.execute(*impl_->commandBuffer);
    if (!result) {
        return result;
    }

    const std::string outputName = graph.firstOutputName();
    RenderGraphResource* output = impl_->executor.outputResource(outputName);
    if (output == nullptr || output->texture == nullptr || impl_->readbackBuffer == nullptr) {
        return makeError(Error::InvalidArgument);
    }
    result = impl_->executor.transitionOutput(
        *impl_->commandBuffer,
        outputName,
        ResourceState::TransferSource);
    if (!result) {
        return result;
    }
    impl_->commandBuffer->copyTextureToBuffer(TextureBufferCopyDesc{
        .texture = output->texture,
        .buffer = impl_->readbackBuffer.get(),
        .width = newWidth,
        .height = newHeight,
        .depth = 1,
        .mipLevel = 0,
        .baseLayer = 0,
    });

    result = impl_->commandBuffer->end();
    if (!result) {
        return result;
    }

    CommandBuffer* commandBuffers[] = {impl_->commandBuffer.get()};
    result = impl_->graphicsQueue->submit(QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = impl_->fence.get(),
    });
    if (!result) {
        return result;
    }
    result = impl_->fence->wait();
    if (!result) {
        return result;
    }

    impl_->readbackBuffer->invalidate();
    void* mapped = impl_->readbackBuffer->map();
    if (mapped == nullptr) {
        return makeError(Error::Failure);
    }
    const uint64_t byteSize = static_cast<uint64_t>(newWidth) * static_cast<uint64_t>(newHeight) * 4ull;
    std::memcpy(impl_->pixels.data(), mapped, static_cast<size_t>(byteSize));
    impl_->readbackBuffer->unmap();

    impl_->width = newWidth;
    impl_->height = newHeight;
    return {};
}

const std::vector<uint32_t>& RenderGraphPreviewRenderer::pixels() const
{
    return impl_->pixels;
}

uint32_t RenderGraphPreviewRenderer::width() const
{
    return impl_->width;
}

uint32_t RenderGraphPreviewRenderer::height() const
{
    return impl_->height;
}

const std::string& RenderGraphPreviewRenderer::lastLog() const
{
    return impl_->lastLog;
}

std::string serializeRenderGraphToString(const RenderGraph& graph)
{
    nlohmann::json root;
    root["version"] = 1;
    root["name"] = graph.name();
    root["nodes"] = nlohmann::json::array();
    root["edges"] = nlohmann::json::array();
    root["outputs"] = nlohmann::json::array();

    for (const RenderGraphNode& node : graph.nodes()) {
        root["nodes"].push_back({
            {"id", node.id},
            {"name", node.name},
            {"type", node.type},
            {"properties", node.properties},
            {"position", {{"x", node.uiX}, {"y", node.uiY}}},
        });
    }

    for (const RenderGraphEdge& edge : graph.edges()) {
        root["edges"].push_back({
            {"id", edge.id},
            {"src", makeRenderGraphFieldName(edge.srcPass, edge.srcField)},
            {"dst", makeRenderGraphFieldName(edge.dstPass, edge.dstField)},
        });
    }

    for (const RenderGraphOutput& output : graph.outputs()) {
        root["outputs"].push_back(makeRenderGraphFieldName(output.passName, output.fieldName));
    }

    return root.dump(4);
}

bool deserializeRenderGraphFromString(
    const std::string& text,
    RenderGraph& outGraph,
    std::string& outMessage)
{
    try {
        nlohmann::json root = nlohmann::json::parse(text);
        if (!root.is_object() || root.value("version", 0) != 1) {
            outMessage = "Unsupported RenderGraph JSON version";
            return false;
        }

        RenderGraph graph;
        graph.clear();
        graph.name_ = root.value("name", "RenderGraph");
        graph.nodes_.clear();
        graph.edges_.clear();
        graph.outputs_.clear();

        uint32_t maxNodeId = 0;
        uint32_t maxEdgeId = 0;
        for (const nlohmann::json& nodeJson : root.at("nodes")) {
            RenderGraphNode node;
            node.id = nodeJson.value("id", 0u);
            node.name = nodeJson.value("name", "");
            node.type = nodeJson.value("type", "");
            node.properties = nodeJson.value("properties", RenderGraphProperties::object());
            if (nodeJson.contains("position")) {
                node.uiX = nodeJson["position"].value("x", 0.0f);
                node.uiY = nodeJson["position"].value("y", 0.0f);
            }
            maxNodeId = std::max(maxNodeId, node.id);
            graph.nodes_.push_back(std::move(node));
        }

        for (const nlohmann::json& edgeJson : root.value("edges", nlohmann::json::array())) {
            RenderGraphEdge edge;
            edge.id = edgeJson.value("id", 0u);
            const std::string src = edgeJson.value("src", "");
            const std::string dst = edgeJson.value("dst", "");
            if (!splitRenderGraphFieldName(src, edge.srcPass, edge.srcField) ||
                !splitRenderGraphFieldName(dst, edge.dstPass, edge.dstField)) {
                outMessage = "Invalid edge endpoint in RenderGraph JSON";
                return false;
            }
            maxEdgeId = std::max(maxEdgeId, edge.id);
            graph.edges_.push_back(std::move(edge));
        }

        for (const nlohmann::json& outputJson : root.value("outputs", nlohmann::json::array())) {
            std::string passName;
            std::string fieldName;
            if (!splitRenderGraphFieldName(outputJson.get<std::string>(), passName, fieldName)) {
                outMessage = "Invalid graph output in RenderGraph JSON";
                return false;
            }
            graph.outputs_.push_back(RenderGraphOutput{
                .passName = std::move(passName),
                .fieldName = std::move(fieldName),
            });
        }

        graph.nextNodeId_ = maxNodeId + 1;
        graph.nextEdgeId_ = maxEdgeId + 1;
        graph.markDirty();

        std::string validationLog;
        if (!graph.validate(validationLog)) {
            outMessage = validationLog;
            return false;
        }

        outGraph = std::move(graph);
        outMessage = "Loaded RenderGraph";
        return true;
    } catch (const std::exception& exception) {
        outMessage = exception.what();
        return false;
    }
}

bool saveRenderGraphToFile(
    const RenderGraph& graph,
    const std::filesystem::path& path,
    std::string& outMessage)
{
    std::error_code error;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            outMessage = error.message();
            return false;
        }
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        outMessage = "Failed to open RenderGraph file for writing";
        return false;
    }
    file << serializeRenderGraphToString(graph);
    outMessage = std::string("Saved RenderGraph to ") + path.string();
    return true;
}

bool loadRenderGraphFromFile(
    const std::filesystem::path& path,
    RenderGraph& outGraph,
    std::string& outMessage)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        outMessage = "Failed to open RenderGraph file";
        return false;
    }
    std::ostringstream stream;
    stream << file.rdbuf();
    return deserializeRenderGraphFromString(stream.str(), outGraph, outMessage);
}

} // namespace metallic::render
