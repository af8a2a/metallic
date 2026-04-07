#pragma once

#include "cluster_lod_builder.h"
#include "frame_context.h"
#include "gpu_cull_resources.h"
#include "gpu_driven_helpers.h"
#include "rhi_backend.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

class ClusterStreamingService {
public:
    struct DebugStats {
        uint32_t activeResidencyNodeCount = 0;
        uint32_t activeResidencyGroupCount = 0;
        uint32_t lastResidencyRequestCount = 0;
        uint32_t lastUnloadRequestCount = 0;
        uint32_t lastResidencyPromotedCount = 0;
        uint32_t lastResidencyEvictedCount = 0;
        uint32_t lastResidentGroupCount = 0;
        uint32_t lastAlwaysResidentGroupCount = 0;
        uint32_t residentHeapCapacity = 0;
        uint32_t residentHeapUsed = 0;
        uint32_t dynamicResidentGroupCount = 0;
        uint32_t pendingResidencyGroupCount = 0;
        uint32_t pendingUnloadGroupCount = 0;
        uint32_t confirmedUnloadGroupCount = 0;
        uint32_t maxLoadsPerFrame = 0;
        uint32_t maxUnloadsPerFrame = 0;
        uint32_t ageThreshold = 0;
        bool resourcesReady = false;
    };

    void setStreamingEnabled(bool enabled) {
        if (m_enableStreaming == enabled) {
            return;
        }

        m_enableStreaming = enabled;
        m_stateDirty = true;
    }

    bool streamingEnabled() const { return m_enableStreaming; }

    void setStreamingBudgetGroups(uint32_t budgetGroups) {
        m_streamingBudgetGroups = budgetGroups;
    }

    uint32_t streamingBudgetGroups() const { return m_streamingBudgetGroups; }

    void setMaxLoadsPerFrame(uint32_t maxLoadsPerFrame) {
        m_maxLoadsPerFrame = std::max(1u, maxLoadsPerFrame);
    }

    uint32_t maxLoadsPerFrame() const { return m_maxLoadsPerFrame; }

    void setMaxUnloadsPerFrame(uint32_t maxUnloadsPerFrame) {
        m_maxUnloadsPerFrame = std::max(1u, maxUnloadsPerFrame);
    }

    uint32_t maxUnloadsPerFrame() const { return m_maxUnloadsPerFrame; }

    void setAgeThreshold(uint32_t ageThreshold) {
        m_ageThreshold = ageThreshold;
    }

    uint32_t ageThreshold() const { return m_ageThreshold; }

    void markStateDirty() { m_stateDirty = true; }

    const DebugStats& debugStats() const { return m_debugStats; }

    bool ready() const {
        for (const FrameBuffers& frameBuffers : m_frameBuffers) {
            if (!frameBuffers.groupResidencyBuffer ||
                !frameBuffers.groupAgeBuffer ||
                !frameBuffers.residencyRequestBuffer ||
                !frameBuffers.residencyRequestStateBuffer ||
                !frameBuffers.unloadRequestBuffer ||
                !frameBuffers.streamingPatchBuffer ||
                !frameBuffers.unloadRequestStateBuffer) {
                return false;
            }
        }
        return m_lodGroupPageTableBuffer && m_residentGroupMeshletIndicesBuffer;
    }

    bool useResidentHeap() const {
        return ready() && m_enableStreaming;
    }

    const RhiBuffer* groupResidencyBuffer() const {
        return activeFrameBuffers().groupResidencyBuffer.get();
    }
    const RhiBuffer* groupAgeBuffer() const {
        return activeFrameBuffers().groupAgeBuffer.get();
    }
    const RhiBuffer* lodGroupPageTableBuffer() const {
        return m_lodGroupPageTableBuffer.get();
    }
    const RhiBuffer* residentGroupMeshletIndicesBuffer() const {
        return m_residentGroupMeshletIndicesBuffer.get();
    }
    const RhiBuffer* residencyRequestBuffer() const {
        return activeFrameBuffers().residencyRequestBuffer.get();
    }
    const RhiBuffer* residencyRequestStateBuffer() const {
        return activeFrameBuffers().residencyRequestStateBuffer.get();
    }
    const RhiBuffer* unloadRequestBuffer() const {
        return activeFrameBuffers().unloadRequestBuffer.get();
    }
    const RhiBuffer* unloadRequestStateBuffer() const {
        return activeFrameBuffers().unloadRequestStateBuffer.get();
    }
    const RhiBuffer* streamingPatchBuffer() const {
        return activeFrameBuffers().streamingPatchBuffer.get();
    }
    uint32_t streamingPatchCount() const {
        return m_framePatchCounts[m_activeFrameSlot % kBufferedFrameCount];
    }

    void runUpdateStage(const ClusterLODData& clusterLodData,
                        const PipelineRuntimeContext& runtimeContext,
                        const FrameContext* frameContext) {
        m_debugStats.activeResidencyNodeCount = clusterLodData.totalNodeCount;
        m_debugStats.activeResidencyGroupCount = clusterLodData.totalGroupCount;
        m_activeFrameSlot = frameContext ? (frameContext->frameIndex % kBufferedFrameCount) : 0u;

        const bool clusterLodAvailable =
            clusterLodData.nodeBuffer.nativeHandle() &&
            clusterLodData.groupBuffer.nativeHandle() &&
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle() &&
            clusterLodData.boundsBuffer.nativeHandle();
        if (!clusterLodAvailable) {
            resetDebugStats();
            return;
        }

        ensureStreamingResources(clusterLodData, runtimeContext);
        if (!ready()) {
            updateDebugStats();
            return;
        }

        const bool sourceBufferChanged =
            m_residencySourceNodeBufferHandle != clusterLodData.nodeBuffer.nativeHandle() ||
            m_residencySourceGroupBufferHandle != clusterLodData.groupBuffer.nativeHandle() ||
            m_residencySourceGroupMeshletIndicesHandle !=
                clusterLodData.groupMeshletIndicesBuffer.nativeHandle();
        if (m_stateDirty || sourceBufferChanged) {
            requestHistoryReset(frameContext);
            rebuildStreamingState(clusterLodData);
        } else {
            runRequestReadbackStage(clusterLodData);
            runResidencyUpdateStage(clusterLodData);
        }

        uploadCanonicalStateToActiveFrame();
        updateDebugStats();
    }

private:
    static constexpr uint32_t kInvalidResidentHeapOffset = UINT32_MAX;
    static constexpr uint32_t kBufferedFrameCount = 2u;

    struct ResidentHeapRange {
        uint32_t offset = 0;
        uint32_t count = 0;
    };

    struct GroupResidentAllocation {
        uint32_t heapOffset = kInvalidResidentHeapOffset;
        uint32_t heapCount = 0;
    };

    struct FrameBuffers {
        std::unique_ptr<RhiBuffer> groupResidencyBuffer;
        std::unique_ptr<RhiBuffer> groupAgeBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestStateBuffer;
        std::unique_ptr<RhiBuffer> unloadRequestBuffer;
        std::unique_ptr<RhiBuffer> unloadRequestStateBuffer;
        std::unique_ptr<RhiBuffer> streamingPatchBuffer;
    };

    FrameBuffers& activeFrameBuffers() {
        return m_frameBuffers[m_activeFrameSlot % kBufferedFrameCount];
    }

    const FrameBuffers& activeFrameBuffers() const {
        return m_frameBuffers[m_activeFrameSlot % kBufferedFrameCount];
    }

    static uint32_t* mappedUint32(RhiBuffer* buffer) {
        if (!buffer || !buffer->mappedData()) {
            return nullptr;
        }
        return static_cast<uint32_t*>(buffer->mappedData());
    }

    static ClusterResidencyRequest* mappedRequests(RhiBuffer* buffer) {
        if (!buffer || !buffer->mappedData()) {
            return nullptr;
        }
        return static_cast<ClusterResidencyRequest*>(buffer->mappedData());
    }

    static StreamingPatch* mappedPatches(RhiBuffer* buffer) {
        if (!buffer || !buffer->mappedData()) {
            return nullptr;
        }
        return static_cast<StreamingPatch*>(buffer->mappedData());
    }

    void resetDebugStats() {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastUnloadRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_debugStats.lastResidentGroupCount = 0;
        m_debugStats.lastAlwaysResidentGroupCount = 0;
        m_debugStats.residentHeapCapacity = 0;
        m_debugStats.residentHeapUsed = 0;
        m_debugStats.dynamicResidentGroupCount = 0;
        m_debugStats.pendingResidencyGroupCount = 0;
        m_debugStats.pendingUnloadGroupCount = 0;
        m_debugStats.confirmedUnloadGroupCount = 0;
        m_debugStats.maxLoadsPerFrame = m_maxLoadsPerFrame;
        m_debugStats.maxUnloadsPerFrame = m_maxUnloadsPerFrame;
        m_debugStats.ageThreshold = m_ageThreshold;
        m_debugStats.resourcesReady = false;
    }

    void requestHistoryReset(const FrameContext* frameContext) const {
        if (!frameContext) {
            return;
        }

        auto* mutableFrameContext = const_cast<FrameContext*>(frameContext);
        mutableFrameContext->historyReset = true;
    }

    void createFrameBufferSet(FrameBuffers& frameBuffers,
                              const PipelineRuntimeContext& runtimeContext,
                              uint32_t frameSlot,
                              uint32_t groupCapacity,
                              uint32_t residentHeapCapacity) {
        RhiBufferDesc residencyDesc{};
        residencyDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        residencyDesc.hostVisible = true;
        const std::string residencyName = "ClusterLodGroupResidency[" + std::to_string(frameSlot) + "]";
        residencyDesc.debugName = residencyName.c_str();
        frameBuffers.groupResidencyBuffer = runtimeContext.resourceFactory->createBuffer(residencyDesc);

        RhiBufferDesc ageDesc{};
        ageDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        ageDesc.hostVisible = true;
        const std::string ageName = "ClusterLodGroupAge[" + std::to_string(frameSlot) + "]";
        ageDesc.debugName = ageName.c_str();
        frameBuffers.groupAgeBuffer = runtimeContext.resourceFactory->createBuffer(ageDesc);

        RhiBufferDesc requestDesc{};
        requestDesc.size = size_t(groupCapacity) * sizeof(ClusterResidencyRequest);
        requestDesc.hostVisible = true;
        const std::string requestName = "ClusterLodResidencyRequests[" + std::to_string(frameSlot) + "]";
        requestDesc.debugName = requestName.c_str();
        frameBuffers.residencyRequestBuffer = runtimeContext.resourceFactory->createBuffer(requestDesc);

        RhiBufferDesc requestStateDesc{};
        requestStateDesc.size = GpuDriven::ComputeDispatchCommandLayout::kBufferSize;
        requestStateDesc.hostVisible = true;
        const std::string requestStateName =
            "ClusterLodResidencyRequestState[" + std::to_string(frameSlot) + "]";
        requestStateDesc.debugName = requestStateName.c_str();
        frameBuffers.residencyRequestStateBuffer =
            runtimeContext.resourceFactory->createBuffer(requestStateDesc);

        RhiBufferDesc unloadDesc{};
        unloadDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        unloadDesc.hostVisible = true;
        const std::string unloadName = "ClusterLodUnloadRequests[" + std::to_string(frameSlot) + "]";
        unloadDesc.debugName = unloadName.c_str();
        frameBuffers.unloadRequestBuffer = runtimeContext.resourceFactory->createBuffer(unloadDesc);

        RhiBufferDesc unloadStateDesc{};
        unloadStateDesc.size = GpuDriven::ComputeDispatchCommandLayout::kBufferSize;
        unloadStateDesc.hostVisible = true;
        const std::string unloadStateName =
            "ClusterLodUnloadRequestState[" + std::to_string(frameSlot) + "]";
        unloadStateDesc.debugName = unloadStateName.c_str();
        frameBuffers.unloadRequestStateBuffer =
            runtimeContext.resourceFactory->createBuffer(unloadStateDesc);

        RhiBufferDesc patchDesc{};
        patchDesc.size = size_t(groupCapacity) * sizeof(StreamingPatch);
        patchDesc.hostVisible = true;
        const std::string patchName = "ClusterLodStreamingPatches[" + std::to_string(frameSlot) + "]";
        patchDesc.debugName = patchName.c_str();
        frameBuffers.streamingPatchBuffer = runtimeContext.resourceFactory->createBuffer(patchDesc);
    }

    void ensureStreamingResources(const ClusterLODData& clusterLodData,
                                  const PipelineRuntimeContext& runtimeContext) {
        if (!runtimeContext.resourceFactory) {
            return;
        }

        const uint32_t groupCapacity = std::max(1u, clusterLodData.totalGroupCount);
        const uint32_t residentHeapCapacity =
            std::max<uint32_t>(1u, static_cast<uint32_t>(clusterLodData.groupMeshletIndices.size()));
        const bool needsRecreate =
            !ready() ||
            m_residencyGroupCapacity != groupCapacity ||
            m_residentHeapCapacity != residentHeapCapacity;
        if (!needsRecreate) {
            return;
        }

        for (uint32_t frameSlot = 0; frameSlot < kBufferedFrameCount; ++frameSlot) {
            m_frameBuffers[frameSlot] = {};
            createFrameBufferSet(m_frameBuffers[frameSlot],
                                 runtimeContext,
                                 frameSlot,
                                 groupCapacity,
                                 residentHeapCapacity);
        }

        RhiBufferDesc groupPageTableDesc{};
        groupPageTableDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        groupPageTableDesc.hostVisible = true;
        groupPageTableDesc.debugName = "ClusterLodGroupPageTable";
        m_lodGroupPageTableBuffer = runtimeContext.resourceFactory->createBuffer(groupPageTableDesc);

        RhiBufferDesc residentHeapDesc{};
        residentHeapDesc.size = size_t(residentHeapCapacity) * sizeof(uint32_t);
        residentHeapDesc.hostVisible = true;
        residentHeapDesc.debugName = "ClusterLodResidentGroupMeshletHeap";
        m_residentGroupMeshletIndicesBuffer =
            runtimeContext.resourceFactory->createBuffer(residentHeapDesc);

        m_residencyGroupCapacity = groupCapacity;
        m_residentHeapCapacity = residentHeapCapacity;
        m_groupResidencyState.assign(groupCapacity, 0u);
        m_groupAgeState.assign(groupCapacity, 0u);
        m_groupPageTableState.assign(groupCapacity, kClusterLodGroupPageInvalidAddress);
        m_residentHeapState.assign(residentHeapCapacity, 0u);
        m_groupPendingUnloadState.assign(groupCapacity, 0u);
        m_unloadRequestSeenScratch.assign(groupCapacity, 0u);
        m_stateDirty = true;
        m_dynamicResidentGroups.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        m_confirmedUnloadGroups.clear();
        m_pendingStreamingPatches.clear();
        m_framePatchCounts.fill(0u);
        m_residencySourceNodeBufferHandle = nullptr;
        m_residencySourceGroupBufferHandle = nullptr;
        m_residencySourceGroupMeshletIndicesHandle = nullptr;
        resetResidentHeapAllocator();
        resetSharedSceneBuffers();
        uploadCanonicalStateToAllFrames();
        updateDebugStats();
    }

    void resetSharedSceneBuffers() {
        if (uint32_t* pageTable = mappedUint32(m_lodGroupPageTableBuffer.get())) {
            std::fill(pageTable,
                      pageTable + m_residencyGroupCapacity,
                      kClusterLodGroupPageInvalidAddress);
        }
        if (uint32_t* residentHeap = mappedUint32(m_residentGroupMeshletIndicesBuffer.get())) {
            std::fill(residentHeap,
                      residentHeap + m_residentHeapCapacity,
                      0u);
        }
    }

    void seedResidencyRequestQueue(FrameBuffers& frameBuffers) {
        if (!frameBuffers.residencyRequestStateBuffer) {
            return;
        }

        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            frameBuffers.residencyRequestStateBuffer.get());
    }

    void seedUnloadRequestQueue(FrameBuffers& frameBuffers) {
        if (!frameBuffers.unloadRequestStateBuffer) {
            return;
        }

        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            frameBuffers.unloadRequestStateBuffer.get());
    }

    void uploadCanonicalStateToFrame(FrameBuffers& frameBuffers) {
        if (uint32_t* words = mappedUint32(frameBuffers.groupResidencyBuffer.get())) {
            std::memcpy(words,
                        m_groupResidencyState.data(),
                        size_t(m_residencyGroupCapacity) * sizeof(uint32_t));
        }
        if (uint32_t* ages = mappedUint32(frameBuffers.groupAgeBuffer.get())) {
            std::memcpy(ages,
                        m_groupAgeState.data(),
                        size_t(m_residencyGroupCapacity) * sizeof(uint32_t));
        }
        if (frameBuffers.residencyRequestBuffer && frameBuffers.residencyRequestBuffer->mappedData()) {
            std::memset(frameBuffers.residencyRequestBuffer->mappedData(),
                        0,
                        frameBuffers.residencyRequestBuffer->size());
        }
        if (frameBuffers.unloadRequestBuffer && frameBuffers.unloadRequestBuffer->mappedData()) {
            std::memset(frameBuffers.unloadRequestBuffer->mappedData(),
                        0,
                        frameBuffers.unloadRequestBuffer->size());
        }
        seedResidencyRequestQueue(frameBuffers);
        seedUnloadRequestQueue(frameBuffers);
    }

    void uploadPatchesToActiveFrame() {
        FrameBuffers& frameBuffers = activeFrameBuffers();
        const uint32_t patchCount =
            std::min<uint32_t>(static_cast<uint32_t>(m_pendingStreamingPatches.size()),
                               m_residencyGroupCapacity);
        m_framePatchCounts[m_activeFrameSlot % kBufferedFrameCount] = patchCount;
        if (patchCount == 0u) {
            return;
        }

        StreamingPatch* patches = mappedPatches(frameBuffers.streamingPatchBuffer.get());
        if (!patches) {
            m_framePatchCounts[m_activeFrameSlot % kBufferedFrameCount] = 0u;
            return;
        }

        std::memcpy(patches,
                    m_pendingStreamingPatches.data(),
                    size_t(patchCount) * sizeof(StreamingPatch));
        m_pendingStreamingPatches.clear();
    }

    void uploadCanonicalStateToAllFrames() {
        for (FrameBuffers& frameBuffers : m_frameBuffers) {
            if (!frameBuffers.groupResidencyBuffer) {
                continue;
            }
            uploadCanonicalStateToFrame(frameBuffers);
        }
    }

    void uploadCanonicalStateToActiveFrame() {
        uploadCanonicalStateToFrame(activeFrameBuffers());
        uploadPatchesToActiveFrame();
    }

    void resetResidentHeapAllocator() {
        m_residentHeapFreeRanges.clear();
        if (m_residentHeapCapacity > 0u) {
            m_residentHeapFreeRanges.push_back({0u, m_residentHeapCapacity});
        }

        m_groupResidentAllocations.assign(m_residencyGroupCapacity, {});
    }

    bool isGroupResident(uint32_t groupIndex) const {
        return groupIndex < m_groupResidencyState.size() &&
               (m_groupResidencyState[groupIndex] & kClusterLodGroupResidencyResident) != 0u;
    }

    bool isGroupAlwaysResident(uint32_t groupIndex) const {
        return groupIndex < m_groupResidencyState.size() &&
               (m_groupResidencyState[groupIndex] & kClusterLodGroupResidencyAlwaysResident) != 0u;
    }

    void collectLeafGroupsForNode(uint32_t nodeIndex,
                                  const ClusterLODData& clusterLodData,
                                  std::vector<uint32_t>& outLeafGroups) const {
        if (nodeIndex >= clusterLodData.nodes.size()) {
            return;
        }

        std::vector<uint32_t> nodeStack;
        nodeStack.push_back(nodeIndex);
        while (!nodeStack.empty()) {
            const uint32_t currentNodeIndex = nodeStack.back();
            nodeStack.pop_back();
            if (currentNodeIndex >= clusterLodData.nodes.size()) {
                continue;
            }

            const GPULodNode& node = clusterLodData.nodes[currentNodeIndex];
            if (node.isLeaf != 0u) {
                for (uint32_t childIndex = 0u; childIndex < node.childCount; ++childIndex) {
                    outLeafGroups.push_back(node.childOffset + childIndex);
                }
                continue;
            }

            for (uint32_t childIndex = 0u; childIndex < node.childCount; ++childIndex) {
                nodeStack.push_back(node.childOffset + childIndex);
            }
        }
    }

    bool allocateResidentHeapRange(uint32_t elementCount, uint32_t& outOffset) {
        if (elementCount == 0u) {
            outOffset = 0u;
            return true;
        }

        for (size_t rangeIndex = 0; rangeIndex < m_residentHeapFreeRanges.size(); ++rangeIndex) {
            ResidentHeapRange& range = m_residentHeapFreeRanges[rangeIndex];
            if (range.count < elementCount) {
                continue;
            }

            outOffset = range.offset;
            range.offset += elementCount;
            range.count -= elementCount;
            if (range.count == 0u) {
                m_residentHeapFreeRanges.erase(
                    m_residentHeapFreeRanges.begin() +
                    static_cast<std::vector<ResidentHeapRange>::difference_type>(rangeIndex));
            }
            return true;
        }

        return false;
    }

    void releaseResidentHeapRange(uint32_t offset, uint32_t elementCount) {
        if (elementCount == 0u || offset == kInvalidResidentHeapOffset) {
            return;
        }

        ResidentHeapRange releasedRange{offset, elementCount};
        auto insertIt = std::lower_bound(
            m_residentHeapFreeRanges.begin(),
            m_residentHeapFreeRanges.end(),
            releasedRange.offset,
            [](const ResidentHeapRange& range, uint32_t value) {
                return range.offset < value;
            });
        m_residentHeapFreeRanges.insert(insertIt, releasedRange);

        if (m_residentHeapFreeRanges.empty()) {
            return;
        }

        std::vector<ResidentHeapRange> mergedRanges;
        mergedRanges.reserve(m_residentHeapFreeRanges.size());
        for (const ResidentHeapRange& range : m_residentHeapFreeRanges) {
            if (!mergedRanges.empty()) {
                ResidentHeapRange& previous = mergedRanges.back();
                if (previous.offset + previous.count == range.offset) {
                    previous.count += range.count;
                    continue;
                }
            }
            mergedRanges.push_back(range);
        }

        m_residentHeapFreeRanges = std::move(mergedRanges);
    }

    void appendStreamingPatch(const StreamingPatch& patch) {
        m_pendingStreamingPatches.push_back(patch);
    }

    void queueLoadPatch(uint32_t groupIndex,
                       uint32_t heapOffset,
                       uint32_t clusterStart,
                       uint32_t clusterCount) {
        StreamingPatch patch{};
        patch.groupIndex = groupIndex;
        patch.residentHeapOffset = heapOffset;
        patch.clusterStart = clusterStart;
        patch.clusterCount = clusterCount;
        appendStreamingPatch(patch);
    }

    void queueUnloadPatch(uint32_t groupIndex) {
        StreamingPatch patch{};
        patch.groupIndex = groupIndex;
        patch.residentHeapOffset = kClusterLodGroupPageInvalidAddress;
        patch.clusterStart = 0u;
        patch.clusterCount = 0u;
        appendStreamingPatch(patch);
    }

    bool queueRefreshPatchForResidentGroup(uint32_t groupIndex,
                                           const ClusterLODData& clusterLodData) {
        if (groupIndex >= m_groupResidentAllocations.size() ||
            groupIndex >= clusterLodData.groups.size()) {
            return false;
        }

        if (!ensureResidentHeapSliceForGroup(groupIndex, clusterLodData)) {
            return false;
        }

        const GroupResidentAllocation& allocation = m_groupResidentAllocations[groupIndex];
        if (allocation.heapOffset == kInvalidResidentHeapOffset) {
            return false;
        }

        const GPUClusterGroup& group = clusterLodData.groups[groupIndex];
        queueLoadPatch(groupIndex,
                       allocation.heapOffset,
                       group.clusterStart,
                       group.clusterCount);
        return true;
    }

    bool ensureResidentHeapSliceForGroup(uint32_t groupIndex,
                                         const ClusterLODData& clusterLodData) {
        if (groupIndex >= m_groupResidentAllocations.size() ||
            groupIndex >= clusterLodData.groups.size()) {
            return false;
        }

        GroupResidentAllocation& allocation = m_groupResidentAllocations[groupIndex];
        if (allocation.heapOffset != kInvalidResidentHeapOffset) {
            return true;
        }

        const GPUClusterGroup& group = clusterLodData.groups[groupIndex];
        if (size_t(group.clusterStart) + group.clusterCount > clusterLodData.groupMeshletIndices.size()) {
            return false;
        }

        uint32_t heapOffset = 0u;
        if (!allocateResidentHeapRange(group.clusterCount, heapOffset)) {
            return false;
        }
        if (heapOffset + group.clusterCount > m_residentHeapCapacity) {
            releaseResidentHeapRange(heapOffset, group.clusterCount);
            return false;
        }

        m_groupPageTableState[groupIndex] = heapOffset;
        allocation.heapOffset = heapOffset;
        allocation.heapCount = group.clusterCount;
        return true;
    }

    void invalidateResidentGroup(uint32_t groupIndex) {
        if (groupIndex >= m_groupResidentAllocations.size()) {
            return;
        }

        m_groupPageTableState[groupIndex] = kClusterLodGroupPageInvalidAddress;
        GroupResidentAllocation& allocation = m_groupResidentAllocations[groupIndex];
        releaseResidentHeapRange(allocation.heapOffset, allocation.heapCount);
        allocation = {};
    }

    void clearPendingUnloadCandidate(uint32_t groupIndex) {
        if (groupIndex < m_groupPendingUnloadState.size()) {
            m_groupPendingUnloadState[groupIndex] = 0u;
        }
        m_confirmedUnloadGroups.erase(
            std::remove(m_confirmedUnloadGroups.begin(),
                        m_confirmedUnloadGroups.end(),
                        groupIndex),
            m_confirmedUnloadGroups.end());
    }

    void touchDynamicResidentGroup(uint32_t groupIndex) {
        clearPendingUnloadCandidate(groupIndex);
        auto it = std::find(m_dynamicResidentGroups.begin(), m_dynamicResidentGroups.end(), groupIndex);
        if (it != m_dynamicResidentGroups.end()) {
            m_dynamicResidentGroups.erase(it);
        }
        m_dynamicResidentGroups.push_back(groupIndex);
    }

    void enqueuePendingResidencyGroup(uint32_t groupIndex) {
        clearPendingUnloadCandidate(groupIndex);
        if (groupIndex < m_groupResidencyState.size()) {
            m_groupResidencyState[groupIndex] |= kClusterLodGroupResidencyRequested;
        }
        if (std::find(m_pendingResidencyGroups.begin(), m_pendingResidencyGroups.end(), groupIndex) ==
            m_pendingResidencyGroups.end()) {
            m_pendingResidencyGroups.push_back(groupIndex);
        }
    }

    bool evictResidentGroup(uint32_t groupIndex) {
        if (!isGroupResident(groupIndex) || isGroupAlwaysResident(groupIndex)) {
            return false;
        }

        auto evictIt = std::find(m_dynamicResidentGroups.begin(), m_dynamicResidentGroups.end(), groupIndex);
        if (evictIt != m_dynamicResidentGroups.end()) {
            m_dynamicResidentGroups.erase(evictIt);
        }
        if (groupIndex < m_residencyGroupCapacity) {
            m_groupResidencyState[groupIndex] &= ~(kClusterLodGroupResidencyResident |
                                                   kClusterLodGroupResidencyRequested);
            m_groupAgeState[groupIndex] = 0u;
        }
        if (groupIndex < m_groupPendingUnloadState.size()) {
            m_groupPendingUnloadState[groupIndex] = 0u;
        }
        invalidateResidentGroup(groupIndex);
        queueUnloadPatch(groupIndex);
        ++m_debugStats.lastResidencyEvictedCount;
        return true;
    }

    void promotePendingResidencyGroups(const ClusterLODData& clusterLodData,
                                       uint32_t& remainingLoads) {
        if (remainingLoads == 0u) {
            return;
        }

        size_t pendingIndex = 0;
        while (pendingIndex < m_pendingResidencyGroups.size() && remainingLoads > 0u) {
            const uint32_t groupIndex = m_pendingResidencyGroups[pendingIndex];
            if (groupIndex >= m_residencyGroupCapacity ||
                groupIndex >= clusterLodData.totalGroupCount) {
                m_pendingResidencyGroups.erase(
                    m_pendingResidencyGroups.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if (isGroupResident(groupIndex)) {
                m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyRequested;
                m_groupAgeState[groupIndex] = 0u;
                if (!isGroupAlwaysResident(groupIndex)) {
                    touchDynamicResidentGroup(groupIndex);
                }
                m_pendingResidencyGroups.erase(
                    m_pendingResidencyGroups.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if (m_streamingBudgetGroups == 0u) {
                break;
            }

            if (m_dynamicResidentGroups.size() >= size_t(m_streamingBudgetGroups)) {
                break;
            }

            if (!ensureResidentHeapSliceForGroup(groupIndex, clusterLodData)) {
                break;
            }

            m_groupResidencyState[groupIndex] |= kClusterLodGroupResidencyResident;
            m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyRequested;
            m_groupAgeState[groupIndex] = 0u;
            const GPUClusterGroup& group = clusterLodData.groups[groupIndex];
            queueLoadPatch(groupIndex,
                           m_groupResidentAllocations[groupIndex].heapOffset,
                           group.clusterStart,
                           group.clusterCount);
            touchDynamicResidentGroup(groupIndex);
            ++m_debugStats.lastResidencyPromotedCount;
            --remainingLoads;
            m_pendingResidencyGroups.erase(
                m_pendingResidencyGroups.begin() +
                static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
        }
    }

    void rebuildStreamingState(const ClusterLODData& clusterLodData) {
        std::fill(m_groupResidencyState.begin(), m_groupResidencyState.end(), 0u);
        std::fill(m_groupAgeState.begin(), m_groupAgeState.end(), 0u);
        std::fill(m_groupPageTableState.begin(),
                  m_groupPageTableState.end(),
                  kClusterLodGroupPageInvalidAddress);
        std::fill(m_residentHeapState.begin(), m_residentHeapState.end(), 0u);
        std::fill(m_groupPendingUnloadState.begin(), m_groupPendingUnloadState.end(), 0u);
        std::fill(m_unloadRequestSeenScratch.begin(), m_unloadRequestSeenScratch.end(), 0u);
        resetResidentHeapAllocator();
        m_dynamicResidentGroups.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        m_confirmedUnloadGroups.clear();
        m_pendingStreamingPatches.clear();
        m_framePatchCounts.fill(0u);
        resetDebugStats();
        resetSharedSceneBuffers();

        std::vector<uint32_t> alwaysResidentGroups;
        for (uint32_t lodRootNode : clusterLodData.primitiveGroupLodRoots) {
            if (lodRootNode == UINT32_MAX || lodRootNode >= clusterLodData.nodes.size()) {
                continue;
            }

            const GPULodNode& lodRoot = clusterLodData.nodes[lodRootNode];
            if (lodRoot.childCount == 0u) {
                continue;
            }

            uint32_t alwaysResidentNode = lodRootNode;
            if (lodRoot.isLeaf == 0u) {
                alwaysResidentNode = lodRoot.childOffset + lodRoot.childCount - 1u;
            }

            alwaysResidentGroups.clear();
            collectLeafGroupsForNode(alwaysResidentNode, clusterLodData, alwaysResidentGroups);
            for (uint32_t groupIndex : alwaysResidentGroups) {
                if (groupIndex >= m_residencyGroupCapacity) {
                    continue;
                }
                if (!ensureResidentHeapSliceForGroup(groupIndex, clusterLodData)) {
                    continue;
                }

                m_groupResidencyState[groupIndex] =
                    kClusterLodGroupResidencyResident | kClusterLodGroupResidencyAlwaysResident;
                m_groupAgeState[groupIndex] = 0u;
                queueLoadPatch(groupIndex,
                               m_groupResidentAllocations[groupIndex].heapOffset,
                               clusterLodData.groups[groupIndex].clusterStart,
                               clusterLodData.groups[groupIndex].clusterCount);
            }
        }

        m_residencySourceNodeBufferHandle = clusterLodData.nodeBuffer.nativeHandle();
        m_residencySourceGroupBufferHandle = clusterLodData.groupBuffer.nativeHandle();
        m_residencySourceGroupMeshletIndicesHandle =
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle();
        m_stateDirty = false;
    }

    void syncCanonicalAgeState(const uint32_t* slotAgeWords) {
        if (!slotAgeWords) {
            return;
        }

        for (uint32_t groupIndex = 0; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (!isGroupResident(groupIndex) || isGroupAlwaysResident(groupIndex)) {
                m_groupAgeState[groupIndex] = 0u;
                continue;
            }
            m_groupAgeState[groupIndex] = slotAgeWords[groupIndex];
        }
    }

    void consumeResidentGroupTouches(const uint32_t* slotResidencyWords) {
        if (!slotResidencyWords) {
            return;
        }

        for (uint32_t groupIndex = 0; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            const uint32_t slotState = slotResidencyWords[groupIndex];
            const bool resident = (slotState & kClusterLodGroupResidencyResident) != 0u;
            const bool requested = (slotState & kClusterLodGroupResidencyRequested) != 0u;
            if (!resident || !requested || !isGroupResident(groupIndex)) {
                continue;
            }

            m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyRequested;
            m_groupAgeState[groupIndex] = 0u;
            if (!isGroupAlwaysResident(groupIndex)) {
                touchDynamicResidentGroup(groupIndex);
            }
        }
    }

    void collectUnloadRequestCandidates() {
        for (uint32_t groupIndex = 0; groupIndex < m_residencyGroupCapacity; ++groupIndex) {
            if (!isGroupResident(groupIndex) || isGroupAlwaysResident(groupIndex)) {
                clearPendingUnloadCandidate(groupIndex);
                continue;
            }

            if (m_groupAgeState[groupIndex] == 0u) {
                clearPendingUnloadCandidate(groupIndex);
                continue;
            }

            if (m_unloadRequestSeenScratch[groupIndex] == 0u) {
                clearPendingUnloadCandidate(groupIndex);
                continue;
            }

            if (m_groupPendingUnloadState[groupIndex] != 0u) {
                if (std::find(m_confirmedUnloadGroups.begin(),
                              m_confirmedUnloadGroups.end(),
                              groupIndex) == m_confirmedUnloadGroups.end()) {
                    m_confirmedUnloadGroups.push_back(groupIndex);
                }
            } else {
                m_groupPendingUnloadState[groupIndex] = 1u;
            }
        }
    }

    void runRequestReadbackStage(const ClusterLODData& clusterLodData) {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastUnloadRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_requestReadbackScratch.clear();
        m_unloadRequestReadbackScratch.clear();
        std::fill(m_unloadRequestSeenScratch.begin(), m_unloadRequestSeenScratch.end(), 0u);

        FrameBuffers& frameBuffers = activeFrameBuffers();
        syncCanonicalAgeState(mappedUint32(frameBuffers.groupAgeBuffer.get()));
        consumeResidentGroupTouches(mappedUint32(frameBuffers.groupResidencyBuffer.get()));

        if (frameBuffers.residencyRequestBuffer && frameBuffers.residencyRequestStateBuffer) {
            const uint32_t requestCapacity = static_cast<uint32_t>(
                frameBuffers.residencyRequestBuffer->size() / sizeof(ClusterResidencyRequest));
            const uint32_t requestCount = std::min<uint32_t>(
                GpuDriven::readWorklistWriteCursor<GpuDriven::ComputeDispatchCommandLayout>(
                    frameBuffers.residencyRequestStateBuffer.get()),
                requestCapacity);
            m_debugStats.lastResidencyRequestCount = requestCount;

            ClusterResidencyRequest* requests = mappedRequests(frameBuffers.residencyRequestBuffer.get());
            if (requests && requestCount > 0u) {
                m_requestReadbackScratch.assign(requests, requests + requestCount);
                for (const ClusterResidencyRequest& request : m_requestReadbackScratch) {
                    if (request.targetGroupIndex >= clusterLodData.totalGroupCount) {
                        continue;
                    }
                    if (isGroupAlwaysResident(request.targetGroupIndex)) {
                        continue;
                    }
                    if (isGroupResident(request.targetGroupIndex)) {
                        m_groupResidencyState[request.targetGroupIndex] &=
                            ~kClusterLodGroupResidencyRequested;
                        m_groupAgeState[request.targetGroupIndex] = 0u;
                        queueRefreshPatchForResidentGroup(request.targetGroupIndex,
                                                          clusterLodData);
                        touchDynamicResidentGroup(request.targetGroupIndex);
                        continue;
                    }
                    enqueuePendingResidencyGroup(request.targetGroupIndex);
                }
            }
        }

        if (frameBuffers.unloadRequestBuffer && frameBuffers.unloadRequestStateBuffer) {
            const uint32_t unloadCapacity =
                static_cast<uint32_t>(frameBuffers.unloadRequestBuffer->size() / sizeof(uint32_t));
            const uint32_t unloadCount = std::min<uint32_t>(
                GpuDriven::readWorklistWriteCursor<GpuDriven::ComputeDispatchCommandLayout>(
                    frameBuffers.unloadRequestStateBuffer.get()),
                unloadCapacity);
            m_debugStats.lastUnloadRequestCount = unloadCount;

            uint32_t* unloadRequests = mappedUint32(frameBuffers.unloadRequestBuffer.get());
            if (unloadRequests && unloadCount > 0u) {
                m_unloadRequestReadbackScratch.assign(unloadRequests, unloadRequests + unloadCount);
                for (uint32_t groupIndex : m_unloadRequestReadbackScratch) {
                    if (groupIndex >= clusterLodData.totalGroupCount ||
                        !isGroupResident(groupIndex) ||
                        isGroupAlwaysResident(groupIndex)) {
                        continue;
                    }
                    m_unloadRequestSeenScratch[groupIndex] = 1u;
                }
            }
        }

        collectUnloadRequestCandidates();
    }

    void runResidencyUpdateStage(const ClusterLODData& clusterLodData) {
        if (!m_enableStreaming) {
            return;
        }

        uint32_t remainingLoads = m_maxLoadsPerFrame;
        uint32_t remainingUnloads = m_maxUnloadsPerFrame;

        size_t confirmedIndex = 0;
        while (confirmedIndex < m_confirmedUnloadGroups.size()) {
            const uint32_t groupIndex = m_confirmedUnloadGroups[confirmedIndex];
            if (groupIndex >= m_residencyGroupCapacity ||
                !isGroupResident(groupIndex) ||
                isGroupAlwaysResident(groupIndex)) {
                m_confirmedUnloadGroups.erase(
                    m_confirmedUnloadGroups.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(confirmedIndex));
                continue;
            }

            if (remainingUnloads == 0u) {
                break;
            }

            if (evictResidentGroup(groupIndex)) {
                --remainingUnloads;
            }

            m_confirmedUnloadGroups.erase(
                m_confirmedUnloadGroups.begin() +
                static_cast<std::vector<uint32_t>::difference_type>(confirmedIndex));
        }

        promotePendingResidencyGroups(clusterLodData, remainingLoads);
    }

    uint32_t computeResidentHeapUsed() const {
        uint32_t freeCount = 0u;
        for (const ResidentHeapRange& range : m_residentHeapFreeRanges) {
            freeCount += range.count;
        }
        return m_residentHeapCapacity - std::min(m_residentHeapCapacity, freeCount);
    }

    void updateDebugStats() {
        m_debugStats.lastResidentGroupCount = 0;
        m_debugStats.lastAlwaysResidentGroupCount = 0;
        m_debugStats.residentHeapCapacity = m_residentHeapCapacity;
        m_debugStats.residentHeapUsed = computeResidentHeapUsed();
        m_debugStats.dynamicResidentGroupCount =
            static_cast<uint32_t>(m_dynamicResidentGroups.size());
        m_debugStats.pendingResidencyGroupCount =
            static_cast<uint32_t>(m_pendingResidencyGroups.size());
        m_debugStats.pendingUnloadGroupCount = 0;
        for (uint8_t pendingState : m_groupPendingUnloadState) {
            m_debugStats.pendingUnloadGroupCount += pendingState != 0u ? 1u : 0u;
        }
        m_debugStats.confirmedUnloadGroupCount =
            static_cast<uint32_t>(m_confirmedUnloadGroups.size());
        m_debugStats.maxLoadsPerFrame = m_maxLoadsPerFrame;
        m_debugStats.maxUnloadsPerFrame = m_maxUnloadsPerFrame;
        m_debugStats.ageThreshold = m_ageThreshold;
        m_debugStats.resourcesReady = ready();

        const uint32_t groupCount =
            std::min(m_debugStats.activeResidencyGroupCount, m_residencyGroupCapacity);
        for (uint32_t groupIndex = 0; groupIndex < groupCount; ++groupIndex) {
            const uint32_t state = m_groupResidencyState[groupIndex];
            if ((state & kClusterLodGroupResidencyResident) != 0u) {
                ++m_debugStats.lastResidentGroupCount;
            }
            if ((state & kClusterLodGroupResidencyAlwaysResident) != 0u) {
                ++m_debugStats.lastAlwaysResidentGroupCount;
            }
        }
    }

    bool m_enableStreaming = false;
    bool m_stateDirty = true;
    uint32_t m_activeFrameSlot = 0u;
    uint32_t m_streamingBudgetGroups = 256u;
    uint32_t m_residencyGroupCapacity = 0;
    uint32_t m_residentHeapCapacity = 0;
    uint32_t m_maxLoadsPerFrame = 128u;
    uint32_t m_maxUnloadsPerFrame = 256u;
    uint32_t m_ageThreshold = 16u;
    const void* m_residencySourceNodeBufferHandle = nullptr;
    const void* m_residencySourceGroupBufferHandle = nullptr;
    const void* m_residencySourceGroupMeshletIndicesHandle = nullptr;
    std::array<FrameBuffers, kBufferedFrameCount> m_frameBuffers;
    std::unique_ptr<RhiBuffer> m_lodGroupPageTableBuffer;
    std::unique_ptr<RhiBuffer> m_residentGroupMeshletIndicesBuffer;
    std::array<uint32_t, kBufferedFrameCount> m_framePatchCounts{};
    std::vector<uint32_t> m_groupResidencyState;
    std::vector<uint32_t> m_groupAgeState;
    std::vector<uint32_t> m_groupPageTableState;
    std::vector<uint32_t> m_residentHeapState;
    std::vector<uint8_t> m_groupPendingUnloadState;
    std::vector<uint8_t> m_unloadRequestSeenScratch;
    std::vector<uint32_t> m_dynamicResidentGroups;
    std::vector<uint32_t> m_pendingResidencyGroups;
    std::vector<StreamingPatch> m_pendingStreamingPatches;
    std::vector<ClusterResidencyRequest> m_requestReadbackScratch;
    std::vector<uint32_t> m_unloadRequestReadbackScratch;
    std::vector<uint32_t> m_confirmedUnloadGroups;
    std::vector<ResidentHeapRange> m_residentHeapFreeRanges;
    std::vector<GroupResidentAllocation> m_groupResidentAllocations;
    DebugStats m_debugStats;
};
