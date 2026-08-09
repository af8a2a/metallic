#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphNode.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <filesystem>
#include <string>
#include <vector>

namespace metallic::render {

struct LegacyEnvironmentMigrationResult {
    bool found = false;
    EnvironmentSettings settings;
    std::string selectedNode;
    std::vector<std::string> ignoredNodes;
    std::string warning;
};

LegacyEnvironmentMigrationResult migrateLegacyEnvironmentSettings(
    RenderGraph& graph,
    const std::filesystem::path& relativePathBase = {});

} // namespace metallic::render
