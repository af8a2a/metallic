#pragma once

#include "Runtime/Render/Core/SlangShaderCache.h"

// Experimental batch-only codec. The caller must qualify this immutable proof
// against the complete source epoch before and after the transaction. This is
// never an ordinary disk-cache lookup and cannot accept an arbitrary old proof.
namespace metallic::render::slang_shader_cache {

class PreparedBatchProof
{
public:
    explicit PreparedBatchProof(const ShaderNamespaceProof& source)
        : proof_(source), encoded_(encodeProof(source)) {}

    bool valid() const { return !encoded_.empty(); }

    Bytes encodeItem(uint64_t request, std::span<const uint32_t> spirv) const
    {
        if (!valid() || spirv.size() < 5 || spirv[0] != 0x07230203 || spirv[4] ||
            spirv.size_bytes() > kMaximumFileBytes ||
            encoded_.size() + spirv.size_bytes() > kMaximumFileBytes - kHeaderSize) { return {}; }
        Bytes output{'M', 'T', 'L', 'S', 'P', 'V', '0', '1'};
        output.reserve(kHeaderSize + encoded_.size() + spirv.size_bytes());
        integer(output, kVersion, 4); integer(output, kHeaderSize, 4); integer(output, request, 8);
        integer(output, encoded_.size(), 8); integer(output, spirv.size_bytes(), 8); integer(output, 0, 8);
        output.insert(output.end(), encoded_.begin(), encoded_.end());
        const auto* data = reinterpret_cast<const uint8_t*>(spirv.data());
        output.insert(output.end(), data, data + spirv.size_bytes());
        const auto bodyChecksum = checksum(std::span<const uint8_t>(output).subspan(kHeaderSize));
        for (size_t byte = 0; byte < 8; ++byte) { output[40 + byte] = static_cast<uint8_t>(bodyChecksum >> (8 * byte)); }
        return output;
    }

    bool readback(std::span<const uint8_t> real, uint64_t request,
        std::span<const uint32_t> expectedSpirv) const
    {
        if (!valid() || real.size() < kHeaderSize || real.size() > kMaximumFileBytes ||
            std::memcmp(real.data(), "MTLSPV01", 8)) { return false; }
        Cursor cursor{real, 8};
        uint64_t version, headerSize, actualRequest, proofSize, spirvSize, payloadChecksum;
        if (!cursor.integer(version, 4) || version != kVersion ||
            !cursor.integer(headerSize, 4) || headerSize != kHeaderSize ||
            !cursor.integer(actualRequest, 8) || actualRequest != request ||
            !cursor.integer(proofSize, 8) || proofSize != encoded_.size() ||
            !cursor.integer(spirvSize, 8) || spirvSize != expectedSpirv.size_bytes() ||
            spirvSize < 20 || spirvSize % 4 ||
            !cursor.integer(payloadChecksum, 8) || proofSize > real.size() - kHeaderSize ||
            spirvSize != real.size() - kHeaderSize - proofSize ||
            checksum(real.subspan(kHeaderSize)) != payloadChecksum) { return false; }
        // Comparing exact serialized bytes proves equality with the already
        // well-formed current proof; it does not parse a caller-provided proof.
        if (std::memcmp(real.data() + kHeaderSize, encoded_.data(), encoded_.size())) { return false; }
        if (expectedSpirv.size() < 5 || expectedSpirv[0] != 0x07230203 || expectedSpirv[4]) { return false; }
        return !std::memcmp(real.data() + kHeaderSize + encoded_.size(), expectedSpirv.data(), spirvSize);
    }

private:
    ShaderNamespaceProof proof_;
    Bytes encoded_;
};

} // namespace metallic::render::slang_shader_cache
