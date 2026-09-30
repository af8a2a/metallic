#include "Runtime/Scene/MeshletStreamReferenceCodec.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <thread>

namespace {
using namespace metallic::scene;

std::vector<uint8_t> makeReferencePage(uint32_t attributes = 31, bool float4Positions = false)
{
    MeshletStreamPayloadHeader h;
    h.magic = 0x4d535047u;
    h.version = 4;
    h.clusterCount = 2;
    h.vertexCount = 128;
    h.triangleIndexCount = 2 * 62 * 3;
    h.attributeFlags = attributes;
    h.positionFormat = uint32_t(float4Positions ? MeshletStreamPayloadFormat::Float32x4 : MeshletStreamPayloadFormat::Float32x3);
    h.normalFormat = uint32_t(MeshletStreamPayloadFormat::Float32x4);
    h.tangentFormat = uint32_t(MeshletStreamPayloadFormat::Float32x4);
    h.texcoord0Format = uint32_t(MeshletStreamPayloadFormat::Float32x2);
    uint32_t offset = sizeof(h);
    const auto allocate = [&](uint32_t bytes) { const uint32_t result = offset; offset = (offset + bytes + 15) & ~15u; return result; };
    h.clusterOffsetBytes = allocate(h.clusterCount * sizeof(MeshletStreamPayloadCluster));
    h.positionOffsetBytes = allocate(h.vertexCount * (float4Positions ? 16 : 12));
    h.triangleOffsetBytes = allocate(h.triangleIndexCount);
    if (attributes & kMeshletStreamPayloadAttributeNormal) { h.normalOffsetBytes = allocate(h.vertexCount * 16); }
    if (attributes & kMeshletStreamPayloadAttributeTangent) { h.tangentOffsetBytes = allocate(h.vertexCount * 16); }
    if (attributes & kMeshletStreamPayloadAttributeTexcoord0) { h.texcoord0OffsetBytes = allocate(h.vertexCount * 8); }
    h.materialCount = h.clusterCount;
    h.materialOffsetBytes = allocate(h.materialCount * 4);
    h.payloadByteSize = h.uncompressedPayloadByteSize = offset;
    std::vector<uint8_t> result(offset, 0);
    std::memcpy(result.data(), &h, sizeof(h));
    for (uint32_t c = 0; c < 2; ++c) {
        MeshletStreamPayloadCluster cluster;
        cluster.vertexOffset = c * 64;
        cluster.vertexCount = 64;
        cluster.triangleOffset = c * 62 * 3;
        cluster.triangleCount = 62;
        cluster.boundingSphere[3] = 0.125f;
        std::memcpy(result.data() + h.clusterOffsetBytes + c * sizeof(cluster), &cluster, sizeof(cluster));
        for (uint32_t t = 0; t < 62; ++t) {
            const uint8_t triangle[3] = {0, uint8_t(t + 1), uint8_t(t + 2)};
            std::memcpy(result.data() + h.triangleOffsetBytes + cluster.triangleOffset + t * 3, triangle, 3);
        }
    }
    for (uint32_t i = 0; i < h.vertexCount; ++i) {
        const float position[4] = {float(i) * 0.00123f + 0.987f, -float(i) * 0.01823f - 1.123f, 0.000014123f * float(i + 1), 1};
        const float normal[4] = {0, 0, 1, 0};
        const float angle = float(i) * 0.025f;
        const float tangent[4] = {std::cos(angle), std::sin(angle), 0, i & 1 ? -1.0f : 1.0f};
        const float uv[2] = {float(i) * 0.12438f, float(i) * -0.03823f};
        std::memcpy(result.data() + h.positionOffsetBytes + i * (float4Positions ? 16 : 12), position, float4Positions ? 16 : 12);
        if (attributes & kMeshletStreamPayloadAttributeNormal) { std::memcpy(result.data() + h.normalOffsetBytes + i * 16, normal, 16); }
        if (attributes & kMeshletStreamPayloadAttributeTangent) { std::memcpy(result.data() + h.tangentOffsetBytes + i * 16, tangent, 16); }
        if (attributes & kMeshletStreamPayloadAttributeTexcoord0) { std::memcpy(result.data() + h.texcoord0OffsetBytes + i * 8, uv, 8); }
    }
    return result;
}

void expectRoundTrip(uint32_t attributes, bool float4Positions, bool generalCompression, uint32_t dropBits)
{
    const auto original = makeReferencePage(attributes, float4Positions);
    MeshletStreamPayloadHeader h;
    std::memcpy(&h, original.data(), sizeof(h));
    const MeshletStreamReferenceCodecOptions options{dropBits, dropBits, generalCompression};
    std::vector<uint8_t> stored, decoded, repeated;
    std::string reason;
    ASSERT_TRUE(encodeMeshletStreamReferencePage(original, stored, reason, options)) << reason;
    ASSERT_TRUE(decodeMeshletStreamReferencePage(stored, decoded, reason)) << reason;
    ASSERT_EQ(decoded.size(), original.size());
    EXPECT_EQ(std::memcmp(decoded.data(), original.data(), sizeof(h)), 0);
    EXPECT_EQ(std::memcmp(decoded.data() + h.clusterOffsetBytes, original.data() + h.clusterOffsetBytes, h.clusterCount * sizeof(MeshletStreamPayloadCluster)), 0);
    EXPECT_EQ(std::memcmp(decoded.data() + h.triangleOffsetBytes, original.data() + h.triangleOffsetBytes, h.triangleIndexCount), 0);
    const uint32_t stride = float4Positions ? 16 : 12;
    for (uint32_t i = 0; i < h.vertexCount; ++i) {
        float p[4], n[4] = {}, t[4] = {}, uv[2] = {};
        std::memcpy(p, original.data() + h.positionOffsetBytes + i * stride, stride);
        if (attributes & 2) { std::memcpy(n, original.data() + h.normalOffsetBytes + i * 16, 16); }
        if (attributes & 16) { std::memcpy(t, original.data() + h.tangentOffsetBytes + i * 16, 16); }
        if (attributes & 4) { std::memcpy(uv, original.data() + h.texcoord0OffsetBytes + i * 8, 8); }
        canonicalizeMeshletStreamReferenceVertex(p, attributes & 2 ? n : nullptr,
            attributes & 4 ? uv : nullptr, attributes & 16 ? t : nullptr, options);
        EXPECT_EQ(std::memcmp(decoded.data() + h.positionOffsetBytes + i * stride, p, stride), 0) << i;
        if (attributes & 2) { EXPECT_EQ(std::memcmp(decoded.data() + h.normalOffsetBytes + i * 16, n, 16), 0) << i; }
        if (attributes & 16) { EXPECT_EQ(std::memcmp(decoded.data() + h.tangentOffsetBytes + i * 16, t, 16), 0) << i; }
        if (attributes & 4) { EXPECT_EQ(std::memcmp(decoded.data() + h.texcoord0OffsetBytes + i * 8, uv, 8), 0) << i; }
    }
    ASSERT_TRUE(encodeMeshletStreamReferencePage(original, repeated, reason, options));
    EXPECT_EQ(stored, repeated);
    EXPECT_LT(stored.size(), original.size());
}
} // namespace

TEST(MeshletStreamReferenceCodec, SourceCanonicalAttributesAndExactTopology)
{
    for (bool compressed : {false, true}) {
        for (bool float4 : {false, true}) {
            for (uint32_t attributes : {1u | 8u, 1u | 8u | 2u, 1u | 8u | 16u, 1u | 8u | 2u | 4u, 31u}) {
                expectRoundTrip(attributes, float4, compressed, 7);
            }
        }
    }
}

TEST(MeshletStreamReferenceCodec, ZeroDiscardBitsPreservePositionAndUvBits)
{
    expectRoundTrip(31, false, false, 0);
    expectRoundTrip(31, true, true, 0);
}

TEST(MeshletStreamReferenceCodec, InvalidShadingUsesExactRawFallback)
{
    auto original = makeReferencePage();
    MeshletStreamPayloadHeader h;
    std::memcpy(&h, original.data(), sizeof(h));
    std::memset(original.data() + h.tangentOffsetBytes, 0, 16);
    const float normal[4] = {0.1f, 0.2f, 0.3f, 0.25f};
    std::memcpy(original.data() + h.normalOffsetBytes, normal, 16);
    std::vector<uint8_t> stored, decoded;
    std::string reason;
    ASSERT_TRUE(encodeMeshletStreamReferencePage(original, stored, reason));
    ASSERT_TRUE(decodeMeshletStreamReferencePage(stored, decoded, reason));
    EXPECT_EQ(std::memcmp(decoded.data() + h.normalOffsetBytes, original.data() + h.normalOffsetBytes, 16), 0);
    EXPECT_EQ(std::memcmp(decoded.data() + h.tangentOffsetBytes, original.data() + h.tangentOffsetBytes, 16), 0);
    // A neighboring valid frame still uses exactly the source canonicalizer.
    float n[4], t[4];
    std::memcpy(n, original.data() + h.normalOffsetBytes + 16, 16);
    std::memcpy(t, original.data() + h.tangentOffsetBytes + 16, 16);
    canonicalizeMeshletStreamReferenceVertex(nullptr, n, nullptr, t);
    EXPECT_EQ(std::memcmp(decoded.data() + h.normalOffsetBytes + 16, n, 16), 0);
    EXPECT_EQ(std::memcmp(decoded.data() + h.tangentOffsetBytes + 16, t, 16), 0);
}

TEST(MeshletStreamReferenceCodec, RejectsCorruptEnvelopeAndPayload)
{
    const auto original = makeReferencePage();
    std::vector<uint8_t> stored, decoded;
    std::string reason;
    ASSERT_TRUE(encodeMeshletStreamReferencePage(original, stored, reason));
    for (size_t offset : {size_t(0), size_t(48), size_t(112), size_t(128), stored.size() / 2, stored.size() - 1}) {
        auto corrupt = stored;
        corrupt[offset] ^= 1;
        EXPECT_FALSE(decodeMeshletStreamReferencePage(corrupt, decoded, reason)) << offset;
        EXPECT_TRUE(decoded.empty());
    }
    for (size_t length : {size_t(0), size_t(111), size_t(144), stored.size() - 1}) {
        EXPECT_FALSE(decodeMeshletStreamReferencePage(std::span(stored).first(length), decoded, reason));
    }
}

TEST(MeshletStreamReferenceCodec, RejectsMalformedLayoutAndClusterTopology)
{
    const auto original = makeReferencePage();
    std::vector<uint8_t> stored;
    std::string reason;
    for (uint32_t offset : {uint32_t(0), uint32_t(original.size()), UINT32_MAX}) {
        auto invalid = original;
        std::memcpy(invalid.data() + offsetof(MeshletStreamPayloadHeader, positionOffsetBytes), &offset, 4);
        EXPECT_FALSE(encodeMeshletStreamReferencePage(invalid, stored, reason));
    }
    auto invalid = original;
    MeshletStreamPayloadHeader h;
    std::memcpy(&h, original.data(), sizeof(h));
    invalid[h.triangleOffsetBytes] = 64;
    EXPECT_FALSE(encodeMeshletStreamReferencePage(invalid, stored, reason));
    invalid = original;
    const uint32_t one = 1;
    std::memcpy(invalid.data() + h.clusterOffsetBytes, &one, 4);
    EXPECT_FALSE(encodeMeshletStreamReferencePage(invalid, stored, reason));
    EXPECT_FALSE(encodeMeshletStreamReferencePage(original, stored, reason, {24, 7, false}));
}

TEST(MeshletStreamReferenceCodec, ParallelCodecIsDeterministic)
{
    const auto original = makeReferencePage();
    std::array<std::vector<uint8_t>, 4> stored;
    std::array<bool, 4> success{};
    std::array<std::thread, 4> threads;
    for (size_t i = 0; i < threads.size(); ++i) {
        threads[i] = std::thread([&, i] {
            std::string reason;
            std::vector<uint8_t> decoded;
            success[i] = encodeMeshletStreamReferencePage(original, stored[i], reason, {7, 7, true}) &&
                decodeMeshletStreamReferencePage(stored[i], decoded, reason) && decoded.size() == original.size();
        });
    }
    for (auto& thread : threads) { thread.join(); }
    for (size_t i = 0; i < threads.size(); ++i) { EXPECT_TRUE(success[i]); EXPECT_EQ(stored[i], stored[0]); }
}

TEST(MeshletStreamReferenceCodec, SouthPoleTangentBasisKeepsAngleAndHandedness)
{
    for (float x : {0.0f, 0.00001f, 0.001f, 0.003f, 0.0048f, 0.00491f, 0.005f, 0.01f}) {
        for (float sign : {-1.0f, 1.0f}) {
            float normal[3] = {x, 0, -std::sqrt(1 - x * x)};
            float tangent[4] = {0, 1, 0, sign};
            canonicalizeMeshletStreamReferenceVertex(nullptr, normal, nullptr, tangent);
            const float cosine = tangent[1];
            EXPECT_GT(cosine, std::cos(0.36f * 3.14159265358979323846f / 180)) << x;
            EXPECT_NEAR(normal[0] * tangent[0] + normal[1] * tangent[1] + normal[2] * tangent[2], 0, 0.000001f) << x;
            EXPECT_FLOAT_EQ(tangent[3], sign);
        }
    }
}
