#pragma once

#include "../rhi_backend.h"

std::unique_ptr<RhiContext> createVulkanContext(const RhiCreateInfo& createInfo,
                                                std::string& errorMessage);

