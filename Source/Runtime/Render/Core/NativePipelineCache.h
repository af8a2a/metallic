#pragma once

#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render {

// Independent of driver binary-cache feedback. Attempts count elected factory
// invocations; validation before lookup and explicit-cache calls are excluded.
struct NativePipelineCacheStats {
    uint64_t lookupCount = 0;
    uint64_t readyHitCount = 0;
    uint64_t singleFlightWaitCount = 0;
    uint64_t nativeCreateCount = 0;
    uint64_t failedCreateCount = 0;
    uint64_t evictionCount = 0;
    size_t readyEntryCount = 0;
    size_t residentCount = 0;
    size_t inFlightCount = 0;
    size_t strongCapacity = 0;
};

namespace detail {

enum class NativePipelineType : uint8_t { Compute, Graphics };

struct NativePipelineKey {
    NativePipelineType type;
    uint64_t psoHash;
    std::vector<ShaderSpecializationConstant> specializationConstants;
    bool operator==(const NativePipelineKey&) const = default;
};

struct NativePipelineAcquisition {
    std::shared_ptr<void> owner;
    uint64_t singleFlightWaitTimeNanoseconds = 0;
    bool nativeObjectReused = false;
};

// One instance per Device. Native owners do not own their Device; the cache and
// all external handles obey the ordinary RHI Device lifetime contract.
class NativePipelineCache {
public:
    explicit NativePipelineCache(size_t strongCapacity = 128);
    ~NativePipelineCache();
    NativePipelineCache(const NativePipelineCache&) = delete;
    NativePipelineCache& operator=(const NativePipelineCache&) = delete;

    [[nodiscard]] Result<NativePipelineAcquisition> acquire(NativePipelineKey key,
        const std::function<Result<std::shared_ptr<void>>()>& create);
    // Evicted native owners are destroyed after releasing the cache mutex.
    void setCapacity(size_t strongCapacity);
    NativePipelineCacheStats stats() const;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace detail
} // namespace metallic::render
