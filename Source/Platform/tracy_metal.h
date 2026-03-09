#pragma once

#include <cstdint>

// Tracy Metal GPU profiling bridge.
// TracyMetal.hmm requires ObjC++ and ARC, so we wrap it behind a C++ interface.
// Metal objects cross this boundary as opaque native handles.

// Opaque handle to tracy::MetalCtx
using TracyMetalCtxHandle = void*;

// Opaque handle to a GPU zone scope (tracy::MetalZoneScope allocated on heap)
using TracyMetalGpuZone = void*;

// Source location data matching tracy::SourceLocationData layout.
// Used to create static source locations at call sites.
struct TracyMetalSrcLoc {
    const char* name;
    const char* function;
    const char* file;
    uint32_t line;
    uint32_t color;
};

TracyMetalCtxHandle tracyMetalCreate(void* deviceHandle);
void tracyMetalDestroy(TracyMetalCtxHandle ctx);
void tracyMetalCollect(TracyMetalCtxHandle ctx);

// GPU zone begin/end — srcloc must point to static storage
TracyMetalGpuZone tracyMetalZoneBeginRender(TracyMetalCtxHandle ctx,
                                             void* renderPassDescHandle,
                                             const TracyMetalSrcLoc* srcloc);
TracyMetalGpuZone tracyMetalZoneBeginCompute(TracyMetalCtxHandle ctx,
                                              void* computePassDescHandle,
                                              const TracyMetalSrcLoc* srcloc);
TracyMetalGpuZone tracyMetalZoneBeginBlit(TracyMetalCtxHandle ctx,
                                           void* blitPassDescHandle,
                                           const TracyMetalSrcLoc* srcloc);
void tracyMetalZoneEnd(TracyMetalGpuZone zone);

// Convenience macros that create a static source location at the call site
#define TracyMetalRenderZone(ctx, desc, name) \
    [&]() -> TracyMetalGpuZone { \
        static constexpr TracyMetalSrcLoc srcloc { name, __FUNCTION__, __FILE__, __LINE__, 0 }; \
        return tracyMetalZoneBeginRender(ctx, desc, &srcloc); \
    }()

#define TracyMetalComputeZone(ctx, desc, name) \
    [&]() -> TracyMetalGpuZone { \
        static constexpr TracyMetalSrcLoc srcloc { name, __FUNCTION__, __FILE__, __LINE__, 0 }; \
        return tracyMetalZoneBeginCompute(ctx, desc, &srcloc); \
    }()

#define TracyMetalBlitZone(ctx, desc, name) \
    [&]() -> TracyMetalGpuZone { \
        static constexpr TracyMetalSrcLoc srcloc { name, __FUNCTION__, __FILE__, __LINE__, 0 }; \
        return tracyMetalZoneBeginBlit(ctx, desc, &srcloc); \
    }()
