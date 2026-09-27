#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace metallic::render {

// Slang 2026.18.2 emits typed OpBufferPointerEXT + OpAccessChain. On GB203
// this loses nested struct offsets, corrupting material/texture indices and
// eventually hanging ray queries. Preserve the original layout type explicitly
// with KHR untyped pointers. Flat uint64 arrays retain their entire typed chain:
// mixed 32/64-bit untyped atomics crash the NVIDIA pipeline compiler. This is a
// compile-time layout policy, not a separate resource or binding path.
// Run before publishing the shader cache; mapped binaries remain byte-identical.
inline bool normalizeNativeDescriptorHeapSpirv(
    std::span<const uint32_t> code, std::vector<uint32_t>& output, std::string& error)
{
    enum : uint32_t {
        Capability = 17, TypeInt = 21, TypeRuntimeArray = 29, TypeStruct = 30,
        TypePointer = 32, Constant = 43, Undef = 1, ConstantNull = 46,
        SpecConstant = 50, SpecConstantOp = 52, Function = 54, FunctionCall = 57, Decorate = 71, MemberDecorate = 72,
        Load = 61, ConvertUToAccelerationStructure = 4447,
        Store = 62, CopyMemory = 63, CopyMemorySized = 64, AccessChain = 65,
        InBoundsAccessChain = 66, PtrAccessChain = 67, ArrayLength = 68,
        InBoundsPtrAccessChain = 70, CopyObject = 83, Bitcast = 124, Select = 169, Phi = 245,
        AtomicLoad = 227, AtomicStore = 228, AtomicExchange = 229,
        AtomicCompareExchange = 230, AtomicCompareExchangeWeak = 231,
        AtomicIIncrement = 232, AtomicIDecrement = 233, AtomicXor = 242,
        ReturnValue = 254, TypeUntypedPointer = 4417, UntypedAccessChain = 4419,
        UntypedInBoundsAccessChain = 4420, UntypedArrayLength = 4425,
        BufferPointer = 5119, DescriptorHeap = 5128, UntypedPointers = 4473,
        Uniform = 2, StorageBuffer = 12, Block = 2, ArrayStride = 6, Offset = 35,
    };
    struct PointerType {
        uint32_t storage;
        uint32_t pointee;
    };
    // Bit flags allow forward-edge/loop discovery to converge before rejecting
    // merges between incompatible representations.
    enum PointerPolicy : uint32_t { NormalizeExplicitLayout = 1, PreserveTypedUInt64 = 2 };
    struct PointerInfo {
        PointerType type;
        uint32_t policy;
    };
    struct IntegerType { uint32_t width; uint32_t signedness; };
    auto fail = [&](const char* reason) {
        error = std::string("Native descriptor heap normalization: ") + reason;
        return false;
    };
    if (code.size() < 5 || code[0] != 0x07230203u || code[3] == 0) {
        return fail("invalid SPIR-V header");
    }
    std::vector<size_t> instructions;
    std::unordered_map<uint32_t, PointerType> pointerTypes;
    std::unordered_map<uint32_t, PointerInfo> pointers;
    std::unordered_map<uint32_t, IntegerType> integers;
    std::unordered_map<uint32_t, uint32_t> arrayElements, arrayStrides, soleMembers, firstOffsets;
    std::unordered_set<uint32_t> blocks;
    std::unordered_map<uint32_t, uint32_t> untypedTypes;
    bool native = false, untyped = false;
    size_t functionsBegin = code.size();
    for (size_t offset = 5; offset < code.size();) {
        const uint32_t count = code[offset] >> 16, op = code[offset] & 0xffffu;
        if (count == 0 || count > code.size() - offset) {
            return fail("truncated instruction");
        }
        uint32_t minimum = 1;
        switch (op) {
        case Capability: case TypeStruct: minimum = 2; break;
        case TypeRuntimeArray: case Decorate: minimum = 3; break;
        case TypeInt: case MemberDecorate: minimum = 4; break;
        case Load: case ConvertUToAccelerationStructure: case Bitcast:
        case AccessChain: case InBoundsAccessChain:
        case TypePointer: case BufferPointer: case CopyObject: case Store:
        case FunctionCall: minimum = 4; break;
        case TypeUntypedPointer: case CopyMemory: minimum = 3; break;
        case Function: case ArrayLength: case Phi:
        case UntypedAccessChain: case UntypedInBoundsAccessChain:
        case PtrAccessChain: case InBoundsPtrAccessChain: minimum = 5; break;
        case Select: minimum = 6; break;
        case CopyMemorySized: minimum = 4; break;
        case ReturnValue: minimum = 2; break;
        }
        if (op >= AtomicLoad && op <= AtomicXor) {
            minimum = op == AtomicStore ? 5 :
                (op == AtomicCompareExchange || op == AtomicCompareExchangeWeak) ? 9 :
                (op == AtomicLoad || op == AtomicIIncrement || op == AtomicIDecrement) ? 6 : 7;
        }
        // OpStore has no result type or result id.
        if (op == Store) { minimum = 3; }
        if (count < minimum) { return fail("incomplete instruction operands"); }
        instructions.push_back(offset);
        if (op == Capability) {
            native |= code[offset + 1] == DescriptorHeap;
            untyped |= code[offset + 1] == UntypedPointers;
        } else if (op == TypeInt) {
            integers[code[offset + 1]] = {code[offset + 2], code[offset + 3]};
        } else if (op == TypeRuntimeArray) {
            arrayElements[code[offset + 1]] = code[offset + 2];
        } else if (op == TypeStruct && count == 3) {
            soleMembers[code[offset + 1]] = code[offset + 2];
        } else if (op == Decorate) {
            if (code[offset + 2] == Block && count == 3) { blocks.insert(code[offset + 1]); }
            if (code[offset + 2] == ArrayStride) {
                if (count != 4) { return fail("invalid ArrayStride decoration"); }
                arrayStrides[code[offset + 1]] = code[offset + 3];
            }
        } else if (op == MemberDecorate && code[offset + 3] == Offset) {
            if (count != 5) { return fail("invalid member Offset decoration"); }
            if (code[offset + 2] == 0) { firstOffsets[code[offset + 1]] = code[offset + 4]; }
        } else if (op == TypePointer) {
            pointerTypes[code[offset + 1]] = {code[offset + 2], code[offset + 3]};
        } else if (op == TypeUntypedPointer) {
            untypedTypes[code[offset + 2]] = code[offset + 1];
        } else if (op == Function && functionsBegin == code.size()) {
            functionsBegin = offset;
        }
        offset += count;
    }
    // Reject the unsafe stdlib AS heap lowering even if a new caller bypasses
    // Core's resolver. The backend's AS handle ABI is a 64-bit device address.
    if (native && untypedTypes.contains(0)) {
        std::unordered_set<uint32_t> heapPointers, heapLoads;
        for (size_t offset : instructions) {
            const uint32_t op = code[offset] & 0xffffu;
            if ((op == UntypedAccessChain || op == UntypedInBoundsAccessChain) &&
                code[offset + 1] == untypedTypes.at(0)) {
                heapPointers.insert(code[offset + 2]);
            } else if (op == Load && heapPointers.contains(code[offset + 3])) {
                heapLoads.insert(code[offset + 2]);
            } else if (op == ConvertUToAccelerationStructure && heapLoads.contains(code[offset + 3])) {
                return fail("AS heap-address loads are unsupported; use Metallic::resolveDescriptor for AS device-address handles");
            }
        }
    }
    if (!native) {
        output = std::vector<uint32_t>(code.begin(), code.end());
        error.clear();
        return true;
    }

    // Recognize only the proven layout, independent of shader names and type IDs.
    const auto flatUInt64Block = [&](const PointerType& pointer) {
        if (pointer.storage != StorageBuffer || !blocks.contains(pointer.pointee) ||
            !soleMembers.contains(pointer.pointee) || !firstOffsets.contains(pointer.pointee) ||
            firstOffsets.at(pointer.pointee) != 0) { return false; }
        const uint32_t array = soleMembers.at(pointer.pointee);
        if (!arrayElements.contains(array) || !arrayStrides.contains(array) || arrayStrides.at(array) != 8) {
            return false;
        }
        const auto element = integers.find(arrayElements.at(array));
        return element != integers.end() && element->second.width == 64 && element->second.signedness == 0;
    };
    auto pointerType = [&](uint32_t id, PointerType& result) {
        if (const auto found = pointerTypes.find(id); found != pointerTypes.end()) {
            result = found->second;
            return true;
        }
        for (const auto& [storage, type] : untypedTypes) {
            if (type == id) { result = {storage, 0}; return true; }
        }
        return false;
    };
    for (size_t offset : instructions) {
        if ((code[offset] & 0xffffu) != BufferPointer) { continue; }
        PointerType type{};
        if (!pointerType(code[offset + 1], type) || !untyped ||
            (type.storage != Uniform && type.storage != StorageBuffer)) {
            return fail("buffer pointer requires native heap and KHR untyped pointers");
        }
        pointers[code[offset + 2]] = {type,
            flatUInt64Block(type) ? PreserveTypedUInt64 : NormalizeExplicitLayout};
    }
    // Discover all chains, including already-normalized input, before validation.
    // A policy must reach loop phis and forward edges regardless of instruction order.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t offset : instructions) {
            const uint32_t op = code[offset] & 0xffffu, count = code[offset] >> 16;
            uint32_t policy = 0;
            auto inherit = [&](uint32_t id) {
                if (const auto source = pointers.find(id); source != pointers.end()) { policy |= source->second.policy; }
            };
            if (op == AccessChain || op == InBoundsAccessChain || op == CopyObject) {
                inherit(code[offset + 3]);
            } else if (op == UntypedAccessChain || op == UntypedInBoundsAccessChain) {
                inherit(code[offset + 4]);
            } else if (op == Select) {
                inherit(code[offset + 4]); inherit(code[offset + 5]);
            } else if (op == Phi) {
                if ((count - 3) % 2 != 0) { return fail("invalid phi operands"); }
                for (uint32_t i = 3; i < count; i += 2) { inherit(code[offset + i]); }
            }
            if (policy == 0) { continue; }
            const uint32_t id = code[offset + 2];
            if (auto found = pointers.find(id); found != pointers.end()) {
                const uint32_t combined = found->second.policy | policy;
                changed |= combined != found->second.policy;
                found->second.policy = combined;
            } else {
                PointerType type{};
                if (!pointerType(code[offset + 1], type)) { return fail("missing derived pointer type"); }
                pointers[id] = {type, policy};
                changed = true;
            }
        }
    }
    auto compatible = [&](uint32_t a, uint32_t b, bool samePointee = false) {
        return pointers.contains(a) && pointers.contains(b) &&
            pointers.at(a).type.storage == pointers.at(b).type.storage &&
            pointers.at(a).policy == pointers.at(b).policy &&
            (!samePointee || pointers.at(a).type.pointee == pointers.at(b).type.pointee);
    };
    std::array<bool, 2> required{};
    for (const auto& [id, pointer] : pointers) {
        if (id >= code[3]) { return fail("pointer id exceeds bound"); }
        if (pointer.policy != NormalizeExplicitLayout && pointer.policy != PreserveTypedUInt64) {
            return fail("mixed typed/untyped pointer policy in select or phi");
        }
        if (pointer.policy == PreserveTypedUInt64 && pointer.type.pointee == 0) {
            return fail("typed uint64 chain contains an untyped pointer");
        }
        if (pointer.type.storage != Uniform && pointer.type.storage != StorageBuffer) {
            return fail("derived pointer changed storage class");
        }
        if (pointer.policy == NormalizeExplicitLayout) {
            required[pointer.type.storage == Uniform ? 0 : 1] = true;
        }
    }
    // Only integer value types are needed for OpAtomicStore, which has no result
    // type. Other atomics carry their scalar result type directly.
    std::unordered_map<uint32_t, uint32_t> integerValues;
    for (size_t offset : instructions) {
        const uint32_t op = code[offset] & 0xffffu, count = code[offset] >> 16;
        if (count >= 3 && integers.contains(code[offset + 1]) &&
            (offset >= functionsBegin || op == Constant || op == SpecConstant || op == SpecConstantOp || op == ConstantNull || op == Undef)) {
            integerValues[code[offset + 2]] = code[offset + 1];
        }
    }
    bool untypedAtomic32 = false, untypedAtomic64 = false;
    for (size_t offset : instructions) {
        const uint32_t op = code[offset] & 0xffffu, count = code[offset] >> 16;
        if ((op == AccessChain || op == InBoundsAccessChain || op == CopyObject ||
             op == UntypedAccessChain || op == UntypedInBoundsAccessChain) && pointers.contains(code[offset + 2])) {
            const uint32_t base = code[offset + ((op == UntypedAccessChain || op == UntypedInBoundsAccessChain) ? 4 : 3)];
            if (!compatible(code[offset + 2], base, op == CopyObject)) { return fail("incompatible derived pointer"); }
        }
        if (op == Select && pointers.contains(code[offset + 2]) &&
            (!compatible(code[offset + 2], code[offset + 4], true) ||
             !compatible(code[offset + 2], code[offset + 5], true))) {
            return fail("mixed pointer select");
        }
        if (op == Phi && pointers.contains(code[offset + 2])) {
            for (uint32_t i = 3; i < count; i += 2) {
                if (!compatible(code[offset + 2], code[offset + i], true)) { return fail("mixed pointer phi"); }
            }
        }
        if (op == FunctionCall) {
            for (uint32_t i = 4; i < count; ++i) {
                if (pointers.contains(code[offset + i])) { return fail("buffer pointer escaped into a function"); }
            }
        }
        if ((op == ReturnValue && pointers.contains(code[offset + 1])) ||
            (op == Store && pointers.contains(code[offset + 2])) ||
            ((op == CopyMemory || op == CopyMemorySized) &&
             (pointers.contains(code[offset + 1]) || pointers.contains(code[offset + 2]))) ||
            ((op == PtrAccessChain || op == InBoundsPtrAccessChain || op == Bitcast) && pointers.contains(code[offset + 3]))) {
            return fail("unsupported buffer pointer escape or memory copy");
        }
        if (op >= AtomicLoad && op <= AtomicXor) {
            const uint32_t target = code[offset + (op == AtomicStore ? 1 : 3)];
            const auto pointer = pointers.find(target);
            if (pointer == pointers.end() || pointer->second.policy != NormalizeExplicitLayout ||
                pointer->second.type.storage != StorageBuffer) { continue; }
            uint32_t scalarType = code[offset + 1];
            if (op == AtomicStore) {
                const auto value = integerValues.find(code[offset + 4]);
                if (value == integerValues.end()) { return fail("unknown atomic store value type"); }
                scalarType = value->second;
            }
            const auto integer = integers.find(scalarType);
            if (integer == integers.end()) { return fail("unknown integer atomic type"); }
            untypedAtomic32 |= integer->second.width == 32;
            untypedAtomic64 |= integer->second.width == 64;
        }
    }
    if (untypedAtomic32 && untypedAtomic64) {
        return fail("mixed 32/64-bit untyped buffer atomics are unsupported; place uint64 atomics in a flat RWStructuredBuffer<uint64_t>");
    }
    if (pointers.empty()) {
        output = std::vector<uint32_t>(code.begin(), code.end());
        error.clear();
        return true;
    }
    if (functionsBegin == code.size()) { return fail("missing function section"); }
    std::vector<uint32_t> normalized(code.begin(), code.begin() + 5);
    std::vector<uint32_t> declarations;
    constexpr std::array<uint32_t, 2> storageClasses{Uniform, StorageBuffer};
    for (size_t i = 0; i < storageClasses.size(); ++i) {
        const uint32_t storage = storageClasses[i];
        if (required[i] && !untypedTypes.contains(storage)) {
            if (normalized[3] == std::numeric_limits<uint32_t>::max()) { return fail("id bound overflow"); }
            const uint32_t id = normalized[3]++;
            untypedTypes[storage] = id;
            declarations.insert(declarations.end(), {(3u << 16) | TypeUntypedPointer, id, storage});
        }
    }
    for (size_t offset : instructions) {
        const uint32_t op = code[offset] & 0xffffu, count = code[offset] >> 16;
        if (offset == functionsBegin) {
            normalized.insert(normalized.end(), declarations.begin(), declarations.end());
        }
        std::vector<uint32_t> instruction(code.begin() + offset, code.begin() + offset + count);
        if ((op == BufferPointer || op == AccessChain || op == InBoundsAccessChain ||
             op == CopyObject || op == Select || op == Phi) && pointers.contains(instruction[2]) &&
            pointers.at(instruction[2]).policy == NormalizeExplicitLayout) {
            instruction[1] = untypedTypes.at(pointers.at(instruction[2]).type.storage);
            if (op == AccessChain || op == InBoundsAccessChain) {
                instruction.insert(instruction.begin() + 3, pointers.at(instruction[3]).type.pointee);
                instruction[0] = (uint32_t(instruction.size()) << 16) |
                    (op == AccessChain ? UntypedAccessChain : UntypedInBoundsAccessChain);
            }
        } else if (op == ArrayLength && pointers.contains(instruction[3]) &&
                   pointers.at(instruction[3]).policy == NormalizeExplicitLayout) {
            instruction.insert(instruction.begin() + 3, pointers.at(instruction[3]).type.pointee);
            instruction[0] = (uint32_t(instruction.size()) << 16) | UntypedArrayLength;
        }
        normalized.insert(normalized.end(), instruction.begin(), instruction.end());
    }
    output = std::move(normalized);
    error.clear();
    return true;
}

} // namespace metallic::render
