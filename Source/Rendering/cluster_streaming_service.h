#pragma once

#include "cluster_lod_builder.h"
#include "frame_context.h"
#include "gpu_cull_resources.h"
#include "gpu_driven_helpers.h"
#include "rhi_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

class ClusterStreamingService {
public:
    struct DebugStats {
        uint32_t activeResidencyNodeCount = 0;
        uint32_t activeResidencyGroupCount = 0;
        uint32_t lastResidencyRequestCount = 0;
        uint32_t lastResidencyPromotedCount = 0;
        uint32_t lastResidencyEvictedCount = 0;
        uint32_t lastResidentNodeCount = 0;
        uint32_t lastAlwaysResidentNodeCount = 0;
        uint32_t lastResidentGroupCount = 0;
        uint32_t lastAlwaysResidentGroupCount = 0;
        uint32_t residentHeapCapacity = 0;
        uint32_t residentHeapUsed = 0;
        uint32_t dynamicResidentNodeCount = 0;
        uint32_t pendingResidencyNodeCount = 0;
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

    void setStreamingBudgetNodes(uint32_t budgetNodes) {
        m_streamingBudgetNodes = budgetNodes;
    }

    uint32_t streamingBudgetNodes() const { return m_streamingBudgetNodes; }

    void markStateDirty() { m_stateDirty = true; }

    const DebugStats& debugStats() const { return m_debugStats; }

    bool ready() const {
        return m_lodNodeResidencyBuffer &&
               m_lodGroupPageTableBuffer &&
               m_residentGroupMeshletIndicesBuffer &&
               m_residencyRequestBuffer &&
               m_residencyRequestStateBuffer;
    }

    bool useResidentHeap() const {
        return ready() && m_enableStreaming;
    }

    const RhiBuffer* lodNodeResidencyBuffer() const { return m_lodNodeResidencyBuffer.get(); }
    const RhiBuffer* lodGroupPageTableBuffer() const { return m_lodGroupPageTableBuffer.get(); }
    const RhiBuffer* residentGroupMeshletIndicesBuffer() const {
        return m_residentGroupMeshletIndicesBuffer.get();
    }
    const RhiBuffer* residencyRequestBuffer() const { return m_residencyRequestBuffer.get(); }
    const RhiBuffer* residencyRequestStateBuffer() const { return m_residencyRequestStateBuffer.get(); }

    void runUpdateStage(const ClusterLODData& clusterLodData,
                        const PipelineRuntimeContext& runtimeContext,
                        const FrameContext* frameContext) {
        m_debugStats.activeResidencyNodeCount = clusterLodData.totalNodeCount;
        m_debugStats.activeResidencyGroupCount = clusterLodData.totalGroupCount;

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
            m_residencySourceGroupMeshletIndicesHandle !=
                clusterLodData.groupMeshletIndicesBuffer.nativeHandle();
        if (m_stateDirty || sourceBufferChanged) {
            requestHistoryReset(frameContext);
            rebuildStreamingState(clusterLodData);
        }

        runRequestReadbackStage(clusterLodData);
        runResidencyUpdateStage(clusterLodData);
        finalizeRequestStage();
        updateDebugStats();
    }

private:
    static constexpr uint32_t kInvalidResidentHeapOffset = UINT32_MAX;

    struct ResidentHeapRange {
        uint32_t offset = 0;
        uint32_t count = 0;
    };

    struct NodeResidentAllocation {
        uint32_t heapOffset = kInvalidResidentHeapOffset;
        uint32_t heapCount = 0;
    };

    void resetDebugStats() {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_debugStats.lastResidentNodeCount = 0;
        m_debugStats.lastAlwaysResidentNodeCount = 0;
        m_debugStats.lastResidentGroupCount = 0;
        m_debugStats.lastAlwaysResidentGroupCount = 0;
        m_debugStats.residentHeapCapacity = 0;
        m_debugStats.residentHeapUsed = 0;
        m_debugStats.dynamicResidentNodeCount = 0;
        m_debugStats.pendingResidencyNodeCount = 0;
        m_debugStats.resourcesReady = false;
    }

    void requestHistoryReset(const FrameContext* frameContext) const {
        if (!frameContext) {
            return;
        }

        auto* mutableFrameContext = const_cast<FrameContext*>(frameContext);
        mutableFrameContext->historyReset = true;
    }

    void ensureStreamingResources(const ClusterLODData& clusterLodData,
                                  const PipelineRuntimeContext& runtimeContext) {
        if (!runtimeContext.resourceFactory) {
            return;
        }

        const uint32_t nodeCapacity = std::max(1u, clusterLodData.totalNodeCount);
        const uint32_t groupCapacity = std::max(1u, clusterLodData.totalGroupCount);
        const uint32_t residentHeapCapacity =
            std::max<uint32_t>(1u, static_cast<uint32_t>(clusterLodData.groupMeshletIndices.size()));
        const bool needsRecreate =
            !m_lodNodeResidencyBuffer ||
            !m_lodGroupPageTableBuffer ||
            !m_residentGroupMeshletIndicesBuffer ||
            !m_residencyRequestBuffer ||
            !m_residencyRequestStateBuffer ||
            m_residencyNodeCapacity != nodeCapacity ||
            m_residencyGroupCapacity != groupCapacity ||
            m_residentHeapCapacity != residentHeapCapacity;
        if (!needsRecreate) {
            return;
        }

        RhiBufferDesc residencyDesc{};
        residencyDesc.size = size_t(nodeCapacity) * sizeof(uint32_t);
        residencyDesc.hostVisible = true;
        residencyDesc.debugName = "ClusterLodNodeResidency";
        m_lodNodeResidencyBuffer = runtimeContext.resourceFactory->createBuffer(residencyDesc);

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

        RhiBufferDesc requestDesc{};
        requestDesc.size = size_t(nodeCapacity) * sizeof(ClusterResidencyRequest);
        requestDesc.hostVisible = true;
        requestDesc.debugName = "ClusterLodResidencyRequests";
        m_residencyRequestBuffer = runtimeContext.resourceFactory->createBuffer(requestDesc);

        RhiBufferDesc requestStateDesc{};
        requestStateDesc.size = GpuDriven::ComputeDispatchCommandLayout::kBufferSize;
        requestStateDesc.hostVisible = true;
        requestStateDesc.debugName = "ClusterLodResidencyRequestState";
        m_residencyRequestStateBuffer =
            runtimeContext.resourceFactory->createBuffer(requestStateDesc);

        m_residencyNodeCapacity = nodeCapacity;
        m_residencyGroupCapacity = groupCapacity;
        m_residentHeapCapacity = residentHeapCapacity;
        m_stateDirty = true;
        m_dynamicResidentNodes.clear();
        m_pendingResidencyNodes.clear();
        m_requestReadbackScratch.clear();
        m_residencyNodeLeafGroups.clear();
        m_residencySourceNodeBufferHandle = nullptr;
        m_residencySourceGroupMeshletIndicesHandle = nullptr;

        if (m_lodNodeResidencyBuffer && m_lodNodeResidencyBuffer->mappedData()) {
            std::memset(m_lodNodeResidencyBuffer->mappedData(), 0, m_lodNodeResidencyBuffer->size());
        }
        if (m_residentGroupMeshletIndicesBuffer &&
            m_residentGroupMeshletIndicesBuffer->mappedData()) {
            std::memset(m_residentGroupMeshletIndicesBuffer->mappedData(),
                        0,
                        m_residentGroupMeshletIndicesBuffer->size());
        }
        if (m_residencyRequestBuffer && m_residencyRequestBuffer->mappedData()) {
            std::memset(m_residencyRequestBuffer->mappedData(), 0, m_residencyRequestBuffer->size());
        }

        invalidateGroupPageTable();
        resetResidentHeapAllocator();
        seedResidencyRequestQueue();
        updateDebugStats();
    }

    void seedResidencyRequestQueue() {
        if (!m_residencyRequestStateBuffer) {
            return;
        }

        GpuDriven::seedWorklistStateBuffer<GpuDriven::ComputeDispatchCommandLayout>(
            m_residencyRequestStateBuffer.get());
    }

    void finalizeRequestStage() {
        seedResidencyRequestQueue();
    }

    uint32_t* residencyStateWords() {
        if (!m_lodNodeResidencyBuffer || !m_lodNodeResidencyBuffer->mappedData()) {
            return nullptr;
        }
        return static_cast<uint32_t*>(m_lodNodeResidencyBuffer->mappedData());
    }

    const uint32_t* residencyStateWords() const {
        return const_cast<ClusterStreamingService*>(this)->residencyStateWords();
    }

    uint32_t* groupPageTableWords() {
        if (!m_lodGroupPageTableBuffer || !m_lodGroupPageTableBuffer->mappedData()) {
            return nullptr;
        }
        return static_cast<uint32_t*>(m_lodGroupPageTableBuffer->mappedData());
    }

    const uint32_t* groupPageTableWords() const {
        return const_cast<ClusterStreamingService*>(this)->groupPageTableWords();
    }

    uint32_t* residentHeapWords() {
        if (!m_residentGroupMeshletIndicesBuffer ||
            !m_residentGroupMeshletIndicesBuffer->mappedData()) {
            return nullptr;
        }
        return static_cast<uint32_t*>(m_residentGroupMeshletIndicesBuffer->mappedData());
    }

    const uint32_t* residentHeapWords() const {
        return const_cast<ClusterStreamingService*>(this)->residentHeapWords();
    }

    ClusterResidencyRequest* residencyRequests() {
        if (!m_residencyRequestBuffer || !m_residencyRequestBuffer->mappedData()) {
            return nullptr;
        }
        return static_cast<ClusterResidencyRequest*>(m_residencyRequestBuffer->mappedData());
    }

    void invalidateGroupPageTable() {
        uint32_t* pageTable = groupPageTableWords();
        if (!pageTable) {
            return;
        }

        std::fill(pageTable,
                  pageTable + m_residencyGroupCapacity,
                  kClusterLodGroupPageInvalidAddress);
    }

    void resetResidentHeapAllocator() {
        m_residentHeapFreeRanges.clear();
        if (m_residentHeapCapacity > 0u) {
            m_residentHeapFreeRanges.push_back({0u, m_residentHeapCapacity});
        }

        m_nodeResidentAllocations.clear();
        m_nodeResidentAllocations.resize(m_residencyNodeCapacity);
    }

    bool isResidencyNodeResident(uint32_t nodeIndex) const {
        const uint32_t* words = residencyStateWords();
        return words &&
               nodeIndex < m_residencyNodeCapacity &&
               (words[nodeIndex] & kClusterLodNodeResidencyResident) != 0u;
    }

    bool isResidencyNodeAlwaysResident(uint32_t nodeIndex) const {
        const uint32_t* words = residencyStateWords();
        return words &&
               nodeIndex < m_residencyNodeCapacity &&
               (words[nodeIndex] & kClusterLodNodeResidencyAlwaysResident) != 0u;
    }

    void buildResidencyNodeLeafGroupsRecursive(uint32_t nodeIndex,
                                               const ClusterLODData& clusterLodData,
                                               std::vector<uint8_t>& builtNodes) {
        if (nodeIndex >= m_residencyNodeLeafGroups.size() || builtNodes[nodeIndex] != 0u) {
            return;
        }

        builtNodes[nodeIndex] = 1u;
        const GPULodNode& node = clusterLodData.nodes[nodeIndex];
        auto& leafGroups = m_residencyNodeLeafGroups[nodeIndex];
        if (node.isLeaf != 0u) {
            leafGroups.reserve(node.childCount);
            for (uint32_t childIndex = 0; childIndex < node.childCount; ++childIndex) {
                leafGroups.push_back(node.childOffset + childIndex);
            }
            return;
        }

        for (uint32_t childIndex = 0; childIndex < node.childCount; ++childIndex) {
            const uint32_t childNodeIndex = node.childOffset + childIndex;
            if (childNodeIndex >= clusterLodData.nodes.size()) {
                continue;
            }

            buildResidencyNodeLeafGroupsRecursive(childNodeIndex, clusterLodData, builtNodes);
            const auto& childLeafGroups = m_residencyNodeLeafGroups[childNodeIndex];
            leafGroups.insert(leafGroups.end(), childLeafGroups.begin(), childLeafGroups.end());
        }
    }

    void rebuildResidencyNodeLeafGroups(const ClusterLODData& clusterLodData) {
        m_residencyNodeLeafGroups.clear();
        m_residencyNodeLeafGroups.resize(clusterLodData.totalNodeCount);
        std::vector<uint8_t> builtNodes(clusterLodData.totalNodeCount, 0u);
        for (uint32_t nodeIndex = 0; nodeIndex < clusterLodData.totalNodeCount; ++nodeIndex) {
            buildResidencyNodeLeafGroupsRecursive(nodeIndex, clusterLodData, builtNodes);
        }
    }

    uint32_t computeNodeResidentClusterCount(uint32_t nodeIndex,
                                             const ClusterLODData& clusterLodData) const {
        if (nodeIndex >= m_residencyNodeLeafGroups.size()) {
            return 0u;
        }

        uint32_t totalClusterCount = 0u;
        for (uint32_t groupIndex : m_residencyNodeLeafGroups[nodeIndex]) {
            if (groupIndex >= clusterLodData.groups.size()) {
                continue;
            }
            totalClusterCount += clusterLodData.groups[groupIndex].clusterCount;
        }
        return totalClusterCount;
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

    bool uploadResidentGroupsForNode(uint32_t nodeIndex,
                                     const ClusterLODData& clusterLodData,
                                     uint32_t heapOffset) {
        uint32_t* pageTable = groupPageTableWords();
        uint32_t* residentHeap = residentHeapWords();
        if (!pageTable || !residentHeap || nodeIndex >= m_residencyNodeLeafGroups.size() ||
            nodeIndex >= m_nodeResidentAllocations.size()) {
            return false;
        }

        uint32_t cursor = heapOffset;
        for (uint32_t groupIndex : m_residencyNodeLeafGroups[nodeIndex]) {
            if (groupIndex >= m_residencyGroupCapacity ||
                groupIndex >= clusterLodData.groups.size()) {
                continue;
            }

            const GPUClusterGroup& group = clusterLodData.groups[groupIndex];
            const size_t clusterStart = group.clusterStart;
            const size_t clusterCount = group.clusterCount;
            if (clusterStart + clusterCount > clusterLodData.groupMeshletIndices.size() ||
                cursor + group.clusterCount > m_residentHeapCapacity) {
                return false;
            }

            std::memcpy(residentHeap + cursor,
                        clusterLodData.groupMeshletIndices.data() + clusterStart,
                        clusterCount * sizeof(uint32_t));
            pageTable[groupIndex] = cursor;
            cursor += group.clusterCount;
        }

        m_nodeResidentAllocations[nodeIndex].heapOffset = heapOffset;
        m_nodeResidentAllocations[nodeIndex].heapCount = cursor - heapOffset;
        return true;
    }

    bool ensureResidentHeapSliceForNode(uint32_t nodeIndex,
                                        const ClusterLODData& clusterLodData) {
        if (nodeIndex >= m_nodeResidentAllocations.size()) {
            return false;
        }

        NodeResidentAllocation& allocation = m_nodeResidentAllocations[nodeIndex];
        if (allocation.heapOffset != kInvalidResidentHeapOffset) {
            return true;
        }

        const uint32_t clusterCount = computeNodeResidentClusterCount(nodeIndex, clusterLodData);
        uint32_t heapOffset = 0u;
        if (!allocateResidentHeapRange(clusterCount, heapOffset)) {
            return false;
        }

        if (!uploadResidentGroupsForNode(nodeIndex, clusterLodData, heapOffset)) {
            releaseResidentHeapRange(heapOffset, clusterCount);
            allocation = {};
            return false;
        }

        return true;
    }

    void invalidateResidentGroupsForNode(uint32_t nodeIndex,
                                         const ClusterLODData& clusterLodData) {
        uint32_t* pageTable = groupPageTableWords();
        if (!pageTable || nodeIndex >= m_residencyNodeLeafGroups.size() ||
            nodeIndex >= m_nodeResidentAllocations.size()) {
            return;
        }

        for (uint32_t groupIndex : m_residencyNodeLeafGroups[nodeIndex]) {
            if (groupIndex >= m_residencyGroupCapacity ||
                groupIndex >= clusterLodData.groups.size()) {
                continue;
            }

            pageTable[groupIndex] = kClusterLodGroupPageInvalidAddress;
        }

        NodeResidentAllocation& allocation = m_nodeResidentAllocations[nodeIndex];
        releaseResidentHeapRange(allocation.heapOffset, allocation.heapCount);
        allocation = {};
    }

    void touchDynamicResidentNode(uint32_t nodeIndex) {
        auto it = std::find(m_dynamicResidentNodes.begin(), m_dynamicResidentNodes.end(), nodeIndex);
        if (it != m_dynamicResidentNodes.end()) {
            m_dynamicResidentNodes.erase(it);
        }
        m_dynamicResidentNodes.push_back(nodeIndex);
    }

    void enqueuePendingResidencyNode(uint32_t nodeIndex) {
        if (std::find(m_pendingResidencyNodes.begin(), m_pendingResidencyNodes.end(), nodeIndex) ==
            m_pendingResidencyNodes.end()) {
            m_pendingResidencyNodes.push_back(nodeIndex);
        }
    }

    void evictOldestDynamicResidencyNode(const ClusterLODData& clusterLodData) {
        if (m_dynamicResidentNodes.empty()) {
            return;
        }

        uint32_t* words = residencyStateWords();
        if (!words) {
            m_dynamicResidentNodes.clear();
            return;
        }

        const uint32_t nodeIndex = m_dynamicResidentNodes.front();
        m_dynamicResidentNodes.erase(m_dynamicResidentNodes.begin());
        if (nodeIndex < m_residencyNodeCapacity) {
            words[nodeIndex] &= ~(kClusterLodNodeResidencyResident |
                                  kClusterLodNodeResidencyRequested);
        }
        invalidateResidentGroupsForNode(nodeIndex, clusterLodData);
        ++m_debugStats.lastResidencyEvictedCount;
    }

    void promotePendingResidencyNodes(const ClusterLODData& clusterLodData) {
        uint32_t* words = residencyStateWords();
        if (!words) {
            return;
        }

        size_t pendingIndex = 0;
        while (pendingIndex < m_pendingResidencyNodes.size()) {
            const uint32_t nodeIndex = m_pendingResidencyNodes[pendingIndex];
            if (nodeIndex >= m_residencyNodeCapacity) {
                m_pendingResidencyNodes.erase(
                    m_pendingResidencyNodes.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if ((words[nodeIndex] & kClusterLodNodeResidencyResident) != 0u) {
                words[nodeIndex] &= ~kClusterLodNodeResidencyRequested;
                if ((words[nodeIndex] & kClusterLodNodeResidencyAlwaysResident) == 0u) {
                    touchDynamicResidentNode(nodeIndex);
                }
                m_pendingResidencyNodes.erase(
                    m_pendingResidencyNodes.begin() +
                    static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
                continue;
            }

            if (m_streamingBudgetNodes == 0u) {
                break;
            }

            while (m_dynamicResidentNodes.size() >= size_t(m_streamingBudgetNodes) &&
                   !m_dynamicResidentNodes.empty()) {
                evictOldestDynamicResidencyNode(clusterLodData);
            }
            if (m_dynamicResidentNodes.size() >= size_t(m_streamingBudgetNodes)) {
                break;
            }

            while (!ensureResidentHeapSliceForNode(nodeIndex, clusterLodData)) {
                if (m_dynamicResidentNodes.empty()) {
                    return;
                }
                evictOldestDynamicResidencyNode(clusterLodData);
            }

            words[nodeIndex] |= kClusterLodNodeResidencyResident;
            words[nodeIndex] &= ~kClusterLodNodeResidencyRequested;
            touchDynamicResidentNode(nodeIndex);
            ++m_debugStats.lastResidencyPromotedCount;
            m_pendingResidencyNodes.erase(
                m_pendingResidencyNodes.begin() +
                static_cast<std::vector<uint32_t>::difference_type>(pendingIndex));
        }
    }

    void rebuildStreamingState(const ClusterLODData& clusterLodData) {
        uint32_t* words = residencyStateWords();
        if (!words) {
            return;
        }

        std::memset(words, 0, m_lodNodeResidencyBuffer->size());
        invalidateGroupPageTable();
        if (m_residentGroupMeshletIndicesBuffer &&
            m_residentGroupMeshletIndicesBuffer->mappedData()) {
            std::memset(m_residentGroupMeshletIndicesBuffer->mappedData(),
                        0,
                        m_residentGroupMeshletIndicesBuffer->size());
        }
        resetResidentHeapAllocator();
        rebuildResidencyNodeLeafGroups(clusterLodData);
        m_dynamicResidentNodes.clear();
        m_pendingResidencyNodes.clear();
        m_requestReadbackScratch.clear();
        resetDebugStats();

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
            if (alwaysResidentNode >= m_residencyNodeCapacity) {
                continue;
            }

            if (!ensureResidentHeapSliceForNode(alwaysResidentNode, clusterLodData)) {
                continue;
            }

            words[alwaysResidentNode] =
                kClusterLodNodeResidencyResident | kClusterLodNodeResidencyAlwaysResident;
        }

        seedResidencyRequestQueue();
        m_residencySourceNodeBufferHandle = clusterLodData.nodeBuffer.nativeHandle();
        m_residencySourceGroupMeshletIndicesHandle =
            clusterLodData.groupMeshletIndicesBuffer.nativeHandle();
        m_stateDirty = false;
    }

    void runRequestReadbackStage(const ClusterLODData& clusterLodData) {
        m_debugStats.lastResidencyRequestCount = 0;
        m_debugStats.lastResidencyPromotedCount = 0;
        m_debugStats.lastResidencyEvictedCount = 0;
        m_requestReadbackScratch.clear();

        if (!m_residencyRequestBuffer || !m_residencyRequestStateBuffer) {
            return;
        }

        const uint32_t requestCapacity = static_cast<uint32_t>(
            m_residencyRequestBuffer->size() / sizeof(ClusterResidencyRequest));
        const uint32_t requestCount = std::min<uint32_t>(
            GpuDriven::readWorklistWriteCursor<GpuDriven::ComputeDispatchCommandLayout>(
                m_residencyRequestStateBuffer.get()),
            requestCapacity);
        m_debugStats.lastResidencyRequestCount = requestCount;

        ClusterResidencyRequest* requests = residencyRequests();
        if (!requests || requestCount == 0u) {
            return;
        }

        m_requestReadbackScratch.assign(requests, requests + requestCount);
        for (const ClusterResidencyRequest& request : m_requestReadbackScratch) {
            if (request.targetNodeIndex >= clusterLodData.totalNodeCount) {
                continue;
            }
            if (isResidencyNodeAlwaysResident(request.targetNodeIndex)) {
                continue;
            }
            if (isResidencyNodeResident(request.targetNodeIndex)) {
                touchDynamicResidentNode(request.targetNodeIndex);
                continue;
            }
            enqueuePendingResidencyNode(request.targetNodeIndex);
        }
    }

    void runResidencyUpdateStage(const ClusterLODData& clusterLodData) {
        if (!m_enableStreaming) {
            return;
        }

        while (m_dynamicResidentNodes.size() > size_t(m_streamingBudgetNodes)) {
            evictOldestDynamicResidencyNode(clusterLodData);
        }
        promotePendingResidencyNodes(clusterLodData);
    }

    uint32_t computeResidentHeapUsed() const {
        uint32_t freeCount = 0u;
        for (const ResidentHeapRange& range : m_residentHeapFreeRanges) {
            freeCount += range.count;
        }
        return m_residentHeapCapacity - std::min(m_residentHeapCapacity, freeCount);
    }

    void updateDebugStats() {
        const uint32_t* words = residencyStateWords();
        const uint32_t* pageTable = groupPageTableWords();
        m_debugStats.lastResidentNodeCount = 0;
        m_debugStats.lastAlwaysResidentNodeCount = 0;
        m_debugStats.lastResidentGroupCount = 0;
        m_debugStats.lastAlwaysResidentGroupCount = 0;
        m_debugStats.residentHeapCapacity = m_residentHeapCapacity;
        m_debugStats.residentHeapUsed = computeResidentHeapUsed();
        m_debugStats.dynamicResidentNodeCount =
            static_cast<uint32_t>(m_dynamicResidentNodes.size());
        m_debugStats.pendingResidencyNodeCount =
            static_cast<uint32_t>(m_pendingResidencyNodes.size());
        m_debugStats.resourcesReady = ready();
        if (!words) {
            return;
        }

        for (uint32_t nodeIndex = 0; nodeIndex < m_debugStats.activeResidencyNodeCount; ++nodeIndex) {
            const uint32_t state = words[nodeIndex];
            if ((state & kClusterLodNodeResidencyResident) != 0u) {
                ++m_debugStats.lastResidentNodeCount;
            }
            if ((state & kClusterLodNodeResidencyAlwaysResident) != 0u) {
                ++m_debugStats.lastAlwaysResidentNodeCount;
            }
        }

        if (pageTable) {
            for (uint32_t groupIndex = 0; groupIndex < m_debugStats.activeResidencyGroupCount;
                 ++groupIndex) {
                if (pageTable[groupIndex] != kClusterLodGroupPageInvalidAddress) {
                    ++m_debugStats.lastResidentGroupCount;
                }
            }
        }

        for (uint32_t nodeIndex = 0; nodeIndex < m_debugStats.activeResidencyNodeCount; ++nodeIndex) {
            if (!isResidencyNodeAlwaysResident(nodeIndex) ||
                nodeIndex >= m_residencyNodeLeafGroups.size()) {
                continue;
            }

            m_debugStats.lastAlwaysResidentGroupCount +=
                static_cast<uint32_t>(m_residencyNodeLeafGroups[nodeIndex].size());
        }
    }

    bool m_enableStreaming = false;
    bool m_stateDirty = true;
    uint32_t m_streamingBudgetNodes = 64u;
    uint32_t m_residencyNodeCapacity = 0;
    uint32_t m_residencyGroupCapacity = 0;
    uint32_t m_residentHeapCapacity = 0;
    const void* m_residencySourceNodeBufferHandle = nullptr;
    const void* m_residencySourceGroupMeshletIndicesHandle = nullptr;
    std::unique_ptr<RhiBuffer> m_lodNodeResidencyBuffer;
    std::unique_ptr<RhiBuffer> m_lodGroupPageTableBuffer;
    std::unique_ptr<RhiBuffer> m_residentGroupMeshletIndicesBuffer;
    std::unique_ptr<RhiBuffer> m_residencyRequestBuffer;
    std::unique_ptr<RhiBuffer> m_residencyRequestStateBuffer;
    std::vector<uint32_t> m_dynamicResidentNodes;
    std::vector<uint32_t> m_pendingResidencyNodes;
    std::vector<ClusterResidencyRequest> m_requestReadbackScratch;
    std::vector<std::vector<uint32_t>> m_residencyNodeLeafGroups;
    std::vector<ResidentHeapRange> m_residentHeapFreeRanges;
    std::vector<NodeResidentAllocation> m_nodeResidentAllocations;
    DebugStats m_debugStats;
};
