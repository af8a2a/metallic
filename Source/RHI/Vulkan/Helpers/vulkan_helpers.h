#pragma once

#ifdef _WIN32

// Aggregate header for the Vulkan Helpers layer.
// Provides: transient memory pools (upload ring, transient pool, readback heap),
// upload/readback service orchestration, and GPU diagnostics/profiling.

#include "vulkan_transient_allocator.h"
#include "vulkan_upload_service.h"
#include "vulkan_diagnostics.h"

#endif // _WIN32
