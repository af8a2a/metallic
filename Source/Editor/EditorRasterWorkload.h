#pragma once

#include "Runtime/Render/Debug/RenderDebug.h"
#include <array>
#include <cstring>
#include <stdexcept>

namespace metallic {

// Readbacks are retained until an explicit graph drain by the benchmark owner.
// A diagnostic execution is never included in the uninstrumented timing sample.
class RasterWorkloadObserver : public render::IRenderDebugObserver {
public:
    using Json = nlohmann::json;
    bool capture = false;
    bool captureCull = false;
    uint64_t editorFrame = 0;
    Json camera;
    void compiled(Json) override {}
    void beginExecution(render::Device& device, debug::DebugEvidenceStamp, render::RenderSubsystemHost*) override { device_ = &device; }
    void endExecution(bool) override {}
    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t,
        std::string_view pass, std::span<const render::DebugResourceBinding> resources, const Json& values) override
    {
        const bool binHeader = checkpoint == "AfterStreamEarlyBins" || checkpoint == "AfterStreamLateBins";
        const bool cullHeader = captureCull && (checkpoint == "AfterStreamEarlyClusterCull" || checkpoint == "AfterStreamLateClusterCull");
        if (!capture || pass != "VBuffer" || (!binHeader && !cullHeader && checkpoint != "StreamEarlyWorkload" && checkpoint != "StreamLateWorkload")) { return; }
        for (const auto& resource : resources) {
            if (!((binHeader || cullHeader) ? resource.id.ends_with(".clusters") : resource.id.ends_with(".workload"))) { continue; }
            if (copies_.size() >= 4096) { throw std::runtime_error("SW workload snapshot limit exceeded"); }
            Copy copy;
            copy.metadata = {{"frame", editorFrame}, {"phase", checkpoint}, {"camera", camera}, {"shader", values}, {"binHeader", binHeader}, {"cullHeader", cullHeader}};
            if (!device_->createBuffer({.size = 128, .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback}, copy.buffer)) { throw std::runtime_error("SW workload readback allocation failed"); }
            render::BufferBarrierDesc barrier{.buffer = resource.buffer, .before = resource.state, .after = render::ResourceState::TransferSource};
            commands.barrier({.buffers = &barrier, .bufferCount = 1});
            commands.copyBuffer({.source = resource.buffer, .destination = copy.buffer.get(), .size = 128});
            std::swap(barrier.before, barrier.after);
            commands.barrier({.buffers = &barrier, .bufferCount = 1});
            copies_.push_back(std::move(copy));
        }
    }
    Json takeAfterDrain()
    {
        constexpr const char* names[] = {"clusters", "triangles", "uniqueVertices", "nonemptyTriangles", "bboxVisits",
            "coveredSamples", "atomicAttempts", "bboxRows", "emptyTriangles", "sumWaveMaxRows", "activeTriangleLanes",
            "launchedTriangleLanes", "bboxArea1To4", "bboxArea5To16", "maxBboxWidth", "invalidClusters"};
        Json samples = Json::array();
        for (auto& copy : copies_) {
            copy.buffer->invalidate();
            const void* data = copy.buffer->map();
            if (!data) { throw std::runtime_error("SW workload map failed"); }
            std::array<uint64_t, 16> counters{};
            std::memcpy(counters.data(), data, sizeof(counters));
            copy.buffer->unmap();
            auto row = std::move(copy.metadata);
            if (row["cullHeader"].get<bool>()) {
                std::array<uint32_t, 32> words{};
                std::memcpy(words.data(), counters.data(), sizeof(words));
                row["counts"] = {{"exactClusters", words[0]}, {"fastSoftware", words[1]},
                    {"fastHardware", words[2]}, {"candidateOverflow", words[14]}};
            } else if (row["binHeader"].get<bool>()) {
                std::array<uint32_t, 32> words{};
                std::memcpy(words.data(), counters.data(), sizeof(words));
                row["bins"] = {{"softwareClusters", words[4]}, {"hardwareClusters", uint64_t(words[0])+words[1]+words[2]+words[3]},
                    {"candidates", words[12]}, {"capacity", words[5]}};
            } else {
                for (size_t i = 0; i < counters.size(); ++i) { row["counts"][names[i]] = counters[i]; }
            }
            samples.push_back(std::move(row));
        }
        copies_.clear();
        return samples;
    }
private:
    struct Copy { std::unique_ptr<render::Buffer> buffer; Json metadata; };
    render::Device* device_ = nullptr;
    std::vector<Copy> copies_;
};

} // namespace metallic
