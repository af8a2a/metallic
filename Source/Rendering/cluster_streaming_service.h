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
        uint32_t lastResidencyPromotedCount = 0;
        uint32_t lastResidencyEvictedCount = 0;
        uint32_t lastResidentGroupCount = 0;
        uint32_t lastAlwaysResidentGroupCount = 0;
        uint32_t residentHeapCapacity = 0;
        uint32_t residentHeapUsed = 0;
        uint32_t dynamicResidentGroupCount = 0;
        uint32_t pendingResidencyGroupCount = 0;
        uint32_t maxLoadsPerFrame = 0;
        uint32_t maxUnloadsPerFrame = 0;
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

    void markStateDirty() { m_stateDirty = true; }

    const DebugStats& debugStats() const { return m_debugStats; }

    bool ready() const {
        for (const FrameBuffers& frameBuffers : m_frameBuffers) {
            if (!frameBuffers.groupResidencyBuffer ||
                !frameBuffers.lodGroupPageTableBuffer ||
                !frameBuffers.residentGroupMeshletIndicesBuffer ||
                !frameBuffers.residencyRequestBuffer ||
                !frameBuffers.residencyRequestStateBuffer) {
                return false;
            }
        }
        return true;
    }

    bool useResidentHeap() const {
        return ready() && m_enableStreaming;
    }

    const RhiBuffer* groupResidencyBuffer() const {
        return activeFrameBuffers().groupResidencyBuffer.get();
    }
    const RhiBuffer* lodGroupPageTableBuffer() const {
        return activeFrameBuffers().lodGroupPageTableBuffer.get();
    }
    const RhiBuffer* residentGroupMeshletIndicesBuffer() const {
        return activeFrameBuffers().residentGroupMeshletIndicesBuffer.get();
    }
    const RhiBuffer* residencyRequestBuffer() const {
        return activeFrameBuffers().residencyRequestBuffer.get();
    }
    const RhiBuffer* residencyRequestStateBuffer() const {
        return activeFrameBuffers().residencyRequestStateBuffer.get();
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

        ++m_updateSerial;

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
        std::unique_ptr<RhiBuffer> lodGroupPageTableBuffer;
        std::unique_ptr<RhiBuffer> residentGroupMeshletIndicesBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestBuffer;
        std::unique_ptr<RhiBuffer> residencyRequestStateBuffer;
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

    void resetDebugStats() {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_debugStats.lastResidentGroupCount = 0;
        m_debugStats.lastAlwaysResidentGroupCount = 0;
        m_debugStats.residentHeapCapacity = 0;
        m_debugStats.residentHeapUsed = 0;
        m_debugStats.dynamicResidentGroupCount = 0;
        m_debugStats.pendingResidencyGroupCount = 0;
        m_debugStats.maxLoadsPerFrame = m_maxLoadsPerFrame;
        m_debugStats.maxUnloadsPerFrame = m_maxUnloadsPerFrame;
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

        RhiBufferDesc groupPageTableDesc{};
        groupPageTableDesc.size = size_t(groupCapacity) * sizeof(uint32_t);
        groupPageTableDesc.hostVisible = true;
        const std::string pageTableName = "ClusterLodGroupPageTable[" + std::to_string(frameSlot) + "]";
        groupPageTableDesc.debugName = pageTableName.c_str();
        frameBuffers.lodGroupPageTableBuffer =
            runtimeContext.resourceFactory->createBuffer(groupPageTableDesc);

        RhiBufferDesc residentHeapDesc{};
        residentHeapDesc.size = size_t(residentHeapCapacity) * sizeof(uint32_t);
        residentHeapDesc.hostVisible = true;
        const std::string residentHeapName =
            "ClusterLodResidentGroupMeshletHeap[" + std::to_string(frameSlot) + "]";
        residentHeapDesc.debugName = residentHeapName.c_str();
        frameBuffers.residentGroupMeshletIndicesBuffer =
            runtimeContext.resourceFactory->createBuffer(residentHeapDesc);

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

        m_residencyGroupCapacity = groupCapacity;
        m_residentHeapCapacity = residentHeapCapacity;
        m_groupResidencyState.assign(groupCapacity, 0u);
        m_groupPageTableState.assign(groupCapacity, kClusterLodGroupPageInvalidAddress);
        m_residentHeapState.assign(residentHeapCapacity, 0u);
        m_groupLastTouchedUpdate.assign(groupCapacity, 0u);
        m_stateDirty = true;
        m_dynamicResidentGroups.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        m_residencySourceNodeBufferHandle = nullptr;
        m_residencySourceGroupBufferHandle = nullptr;
        m_residencySourceGroupMeshletIndicesHandle = nullptr;
        resetResidentHeapAllocator();
        uploadCanonicalStateToAllFrames();
        updateDebugStats();
    }

    void seedResidencyRequestQueue(FrameBuffers& frameBuffers) {
        if (!frameBuffers.residencyRequestStateBuffer) {
            return;
        }

        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            frameBuffers.residencyRequestStateBuffer.get());
    }

    void uploadCanonicalStateToFrame(FrameBuffers& frameBuffers) {
        if (uint32_t* words = mappedUint32(frameBuffers.groupResidencyBuffer.get())) {
            std::memcpy(words,
                        m_groupResidencyState.data(),
                        size_t(m_residencyGroupCapacity) * sizeof(uint32_t));
        }
        if (uint32_t* pageTable = mappedUint32(frameBuffers.lodGroupPageTableBuffer.get())) {
            std::memcpy(pageTable,
                        m_groupPageTableState.data(),
                        size_t(m_residencyGroupCapacity) * sizeof(uint32_t));
        }
        if (uint32_t* residentHeap = mappedUint32(frameBuffers.residentGroupMeshletIndicesBuffer.get())) {
            std::memcpy(residentHeap,
                        m_residentHeapState.data(),
                        size_t(m_residentHeapCapacity) * sizeof(uint32_t));
        }
        if (frameBuffers.residencyRequestBuffer && frameBuffers.residencyRequestBuffer->mappedData()) {
            std::memset(frameBuffers.residencyRequestBuffer->mappedData(),
                        0,
                        frameBuffers.residencyRequestBuffer->size());
        }
        seedResidencyRequestQueue(frameBuffers);
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
    }

    void resetResidentHeapAllocator() {
        m_residentHeapFreeRanges.clear();
        if (m_residentHeapCapacity > 0u) {
            m_residentHeapFreeRanges.push_back({0u, m_residentHeapCapacity});
        }

        m_groupResidentAllocations.assign(m_residencyGroupCapacity, {});
    }

    uint32_t dynamicResidentHardLimit() const {
        if (m_streamingBudgetGroups == 0u) {
            return 0u;
        }

        const uint32_t headroom = std::max(1u, m_maxLoadsPerFrame * kBufferedFrameCount);
        return std::min(m_residencyGroupCapacity, m_streamingBudgetGroups + headroom);
    }

    bool canEvictDynamicResidentGroup(uint32_t groupIndex) const {
        if (groupIndex >= m_groupLastTouchedUpdate.size()) {
            return true;
        }

        const uint64_t lastTouchedUpdate = m_groupLastTouchedUpdate[groupIndex];
        return m_updateSerial > (lastTouchedUpdate + kBufferedFrameCount);
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

        std::memcpy(m_residentHeapState.data() + heapOffset,
                    clusterLodData.groupMeshletIndices.data() + group.clusterStart,
                    size_t(group.clusterCount) * sizeof(uint32_t));
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

    void touchDynamicResidentGroup(uint32_t groupIndex) {
        auto it = std::find(m_dynamicResidentGroups.begin(), m_dynamicResidentGroups.end(), groupIndex);
        if (it != m_dynamicResidentGroups.end()) {
            m_dynamicResidentGroups.erase(it);
        }
        if (groupIndex < m_groupLastTouchedUpdate.size()) {
            m_groupLastTouchedUpdate[groupIndex] = m_updateSerial;
        }
        m_dynamicResidentGroups.push_back(groupIndex);
    }

    void enqueuePendingResidencyGroup(uint32_t groupIndex) {
        if (groupIndex < m_groupResidencyState.size()) {
            m_groupResidencyState[groupIndex] |= kClusterLodGroupResidencyRequested;
        }
        if (std::find(m_pendingResidencyGroups.begin(), m_pendingResidencyGroups.end(), groupIndex) ==
            m_pendingResidencyGroups.end()) {
            m_pendingResidencyGroups.push_back(groupIndex);
        }
    }

    bool evictOldestDynamicResidentGroup() {
        if (m_dynamicResidentGroups.empty()) {
            return false;
        }

        auto evictIt = m_dynamicResidentGroups.end();
        for (auto it = m_dynamicResidentGroups.begin(); it != m_dynamicResidentGroups.end(); ++it) {
            if (canEvictDynamicResidentGroup(*it)) {
                evictIt = it;
                break;
            }
        }
        if (evictIt == m_dynamicResidentGroups.end()) {
            return false;
        }

        const uint32_t groupIndex = *evictIt;
        m_dynamicResidentGroups.erase(evictIt);
        if (groupIndex < m_residencyGroupCapacity) {
            m_groupResidencyState[groupIndex] &= ~(kClusterLodGroupResidencyResident |
                                                   kClusterLodGroupResidencyRequested);
        }
        invalidateResidentGroup(groupIndex);
        ++m_debugStats.lastResidencyEvictedCount;
        return true;
    }

    void promotePendingResidencyGroups(const ClusterLODData& clusterLodData,
                                       uint32_t& remainingLoads,
                                       uint32_t& remainingUnloads) {
        if (remainingLoads == 0u) {
            return;
        }

        const uint32_t dynamicResidentLimit = dynamicResidentHardLimit();
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

            while (m_dynamicResidentGroups.size() >= size_t(dynamicResidentLimit) &&
                   remainingUnloads > 0u &&
                   !m_dynamicResidentGroups.empty()) {
                if (!evictOldestDynamicResidentGroup()) {
                    break;
                }
                --remainingUnloads;
            }
            if (m_dynamicResidentGroups.size() >= size_t(dynamicResidentLimit)) {
                break;
            }

            while (!ensureResidentHeapSliceForGroup(groupIndex, clusterLodData)) {
                if (remainingUnloads == 0u || m_dynamicResidentGroups.empty()) {
                    return;
                }
                if (!evictOldestDynamicResidentGroup()) {
                    return;
                }
                --remainingUnloads;
            }

            m_groupResidencyState[groupIndex] |= kClusterLodGroupResidencyResident;
            m_groupResidencyState[groupIndex] &= ~kClusterLodGroupResidencyRequested;
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
        std::fill(m_groupPageTableState.begin(),
                  m_groupPageTableState.end(),
                  kClusterLodGroupPageInvalidAddress);
        std::fill(m_residentHeapState.begin(), m_residentHeapState.end(), 0u);
        resetResidentHeapAllocator();
        m_dynamicResidentGroups.clear();
        m_pendingResidencyGroups.clear();
        m_requestReadbackScratch.clear();
        resetDebugStats();

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
            }
        }

        m_residencySourceNodeBufferHandle = clusterLodData.nodeBuffer.nativeHandle();
        m_residencySourceGroupBufferHandle = clusterLodData.groupBuffer.nativeHandle();
        m_residencySourceGroupMeshletIndicesHandle =
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle();
        m_stateDirty = false;
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
            if (!isGroupAlwaysResident(groupIndex)) {
                touchDynamicResidentGroup(groupIndex);
            }
        }
    }

    void runRequestReadbackStage(const ClusterLODData& clusterLodData) {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_requestReadbackScratch.clear();

        FrameBuffers& frameBuffers = activeFrameBuffers();
        consumeResidentGroupTouches(mappedUint32(frameBuffers.groupResidencyBuffer.get()));

        if (!frameBuffers.residencyRequestBuffer || !frameBuffers.residencyRequestStateBuffer) {
            return;
        }

        const uint32_t requestCapacity = static_cast<uint32_t>(
            frameBuffers.residencyRequestBuffer->size() / sizeof(ClusterResidencyRequest));
        const uint32_t requestCount = std::min<uint32_t>(
            GpuDriven::readWorklistWriteCursor<GpuDriven::ComputeDispatchCommandLayout>(
                frameBuffers.residencyRequestStateBuffer.get()),
            requestCapacity);
        m_debugStats.lastResidencyRequestCount = requestCount;

        ClusterResidencyRequest* requests = mappedRequests(frameBuffers.residencyRequestBuffer.get());
        if (!requests || requestCount == 0u) {
            return;
        }

        m_requestReadbackScratch.assign(requests, requests + requestCount);
        for (const ClusterResidencyRequest& request : m_requestReadbackScratch) {
            if (request.targetGroupIndex >= clusterLodData.totalGroupCount) {
                continue;
            }
            if (isGroupAlwaysResident(request.targetGroupIndex)) {
                continue;
            }
            if (isGroupResident(request.targetGroupIndex)) {
                touchDynamicResidentGroup(request.targetGroupIndex);
                continue;
            }
            enqueuePendingResidencyGroup(request.targetGroupIndex);
        }
    }

    void runResidencyUpdateStage(const ClusterLODData& clusterLodData) {
        if (!m_enableStreaming) {
            return;
        }

        uint32_t remainingLoads = m_maxLoadsPerFrame;
        uint32_t remainingUnloads = m_maxUnloadsPerFrame;
        const uint32_t dynamicResidentLimit = dynamicResidentHardLimit();

        while (m_dynamicResidentGroups.size() > size_t(dynamicResidentLimit) &&
               remainingUnloads > 0u) {
            if (!evictOldestDynamicResidentGroup()) {
                break;
            }
            --remainingUnloads;
        }

        promotePendingResidencyGroups(clusterLodData, remainingLoads, remainingUnloads);
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
        m_debugStats.maxLoadsPerFrame = m_maxLoadsPerFrame;
        m_debugStats.maxUnloadsPerFrame = m_maxUnloadsPerFrame;
        m_debugStats.resourcesReady = ready();

        for (uint32_t groupIndex = 0; groupIndex < m_debugStats.activeResidencyGroupCount; ++groupIndex) {
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
    const void* m_residencySourceNodeBufferHandle = nullptr;
    const void* m_residencySourceGroupBufferHandle = nullptr;
    const void* m_residencySourceGroupMeshletIndicesHandle = nullptr;
    uint64_t m_updateSerial = 0;
    std::array<FrameBuffers, kBufferedFrameCount> m_frameBuffers;
    std::vector<uint32_t> m_groupResidencyState;
    std::vector<uint32_t> m_groupPageTableState;
    std::vector<uint32_t> m_residentHeapState;
    std::vector<uint64_t> m_groupLastTouchedUpdate;
    std::vector<uint32_t> m_dynamicResidentGroups;
    std::vector<uint32_t> m_pendingResidencyGroups;
    std::vector<ClusterResidencyRequest> m_requestReadbackScratch;
    std::vector<ResidentHeapRange> m_residentHeapFreeRanges;
    std::vector<GroupResidentAllocation> m_groupResidentAllocations;
    DebugStats m_debugStats;
};
