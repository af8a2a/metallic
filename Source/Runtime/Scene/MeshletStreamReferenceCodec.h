#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace metallic::scene {

struct MeshletStreamReferenceCodecOptions {
    uint32_t discardedPositionBits = 7;
    uint32_t discardedTexcoordBits = 7;
    bool generalCompression = false;
};

// Reference precision: per-cluster float-bit base/deltas for P/UV, and a
// shared 22-bit octahedral normal / 10-bit tangent-frame word for N/T.
// Attribute pointers can be null. The helper must be applied to source
// attributes once, before comparing them to decoded cache attributes.
void canonicalizeMeshletStreamReferenceVertex(float* position3, float* normal3,
    float* texcoord2, float* tangent4,
    const MeshletStreamReferenceCodecOptions& options = {});

bool encodeMeshletStreamReferencePage(std::span<const uint8_t> decoded,
    std::vector<uint8_t>& stored, std::string& reason,
    const MeshletStreamReferenceCodecOptions& options = {});
bool decodeMeshletStreamReferencePage(std::span<const uint8_t> stored,
    std::vector<uint8_t>& decoded, std::string& reason);

} // namespace metallic::scene
