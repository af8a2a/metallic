#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace metallic::render::detail {

TextureUsageBits addTextureUsage(TextureUsageBits usage, TextureUsageBits flag);
BufferUsageBits addBufferUsage(BufferUsageBits usage, BufferUsageBits flag);
bool isTextureField(const RenderGraphField& field);
bool isBufferField(const RenderGraphField& field);
bool isBindlessField(const RenderGraphField& field);
bool isBindlessSampledImageField(const RenderGraphField& field);
bool isBindlessBufferField(const RenderGraphField& field);
bool accessWrites(RenderGraphResourceAccess access);
ResourceState stateForAccess(RenderGraphResourceAccess access);
TextureUsageBits textureUsageForAccess(RenderGraphResourceAccess access);
BufferUsageBits bufferUsageForAccess(RenderGraphResourceAccess access);
BufferViewType bufferViewTypeForField(const RenderGraphField& field);
bool accessMatchesResourceType(RenderGraphResourceAccess access, RenderGraphResourceType resourceType);
TextureUsageBits textureUsageForField(const RenderGraphField& field);
BufferUsageBits bufferUsageForField(const RenderGraphField& field);
void applyAccessDefaults(RenderGraphField& field);
RenderGraphResourceAccess explicitAccessForState(RenderGraphResourceType type, ResourceState state);
Format resolveFormat(Format format, Format defaultFormat);
std::string resultMessage(std::string_view label, const Result& result);
std::string passProfileMarkerName(const std::string& name, const std::string& type);
ColorValue debugLabelColorFromArgb(uint32_t argb);
const char* queueTypeName(QueueType type);
Queue* queueForSubmitDesc(const RenderGraphSubmitDesc& desc, QueueType type);
bool isOutputMarked(const RenderGraph& graph, std::string_view fullName);
bool nodeNameExists(const std::vector<RenderGraphNode>& nodes, std::string_view name, uint32_t ignoreId = 0);
const RenderGraphNode* findNodeByName(const std::vector<RenderGraphNode>& nodes, std::string_view name);
std::string validationPrefix(std::string_view issue);
bool validateAcyclic(
    const std::vector<RenderGraphNode>& nodes,
    const std::vector<RenderGraphEdge>& edges,
    std::string& log);

struct ActiveGraph {
    std::unordered_set<std::string> activePasses;
    std::vector<std::string> executionOrder;
};

bool buildActiveGraph(const RenderGraph& graph, ActiveGraph& activeGraph, std::string& log);

} // namespace metallic::render::detail