#include "RhiTest.h"
#include "json.hpp"
#include <fstream>

namespace metallic::tests {
namespace {
using namespace render;
constexpr uint64_t kMiB = 1024 * 1024;

class MemoryBudgetTest final : public RhiTest {
public:
    MemoryBudgetTest() { type = RhiTestType::Resource; name = "unified_memory_budget"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto& device = context.device;
        const auto initial = device.memoryBudget();
        if (initial.primaryDeviceLocalHeap == UINT32_MAX || initial.availableBytes < 64 * kMiB) {
            return RhiTestResult::skip("Needs a device-local heap with 64 MiB headroom");
        }
        struct Restore {
            Device& device;
            MemoryBudgetPolicy policy;
            ~Restore() { device.setMemoryBudgetPolicy(policy); }
        } restore{device, initial.policy};
        auto policy = initial.policy;
        policy.enabled = true;
        policy.safetyBytes = kMiB;
        policy.deviceLocalHeapLimitBytes = initial.heaps[initial.primaryDeviceLocalHeap].usageBytes + 32 * kMiB;
        device.setMemoryBudgetPolicy(policy);
        MemoryBudgetReservation future;
        if (!device.reserveMemoryBudget(24 * kMiB).transform([&](auto rhiValue) { future = std::move(rhiValue); })) { return RhiTestResult::fail("Cannot reserve test headroom"); }
        MemoryBudgetReservation moved = std::move(future);
        if (future || !moved || device.memoryBudget().reservedBytes != initial.reservedBytes + 24 * kMiB) {
            return RhiTestResult::fail("Reservation move lost or duplicated credits");
        }
        std::unique_ptr<Buffer> rejected;
        auto result = device.createBuffer({.size = 16 * kMiB, .usage = BufferUsageBits::Storage,
            .memoryDomain = MemoryBudgetDomain::Geometry}).transform([&](auto rhiValue) { rejected = std::move(rhiValue); });
        if (!hasError(result, Error::OutOfMemory) || rejected) {
            return RhiTestResult::fail("Geometry consumed reserved feature headroom");
        }
        std::unique_ptr<Texture> rejectedTexture;
        result = device.createTexture({.usage = TextureUsageBits::Sampled, .format = Format::Rgba8Unorm,
            .width = 2048, .height = 2048, .memoryDomain = MemoryBudgetDomain::MaterialTextures}).transform([&](auto rhiValue) { rejectedTexture = std::move(rhiValue); });
        if (!hasError(result, Error::OutOfMemory) || rejectedTexture) {
            return RhiTestResult::fail("Textures bypassed the shared budget");
        }
        moved.reset();
        std::unique_ptr<Buffer> geometry;
        if (!device.createBuffer({.size = 4 * kMiB, .usage = BufferUsageBits::Storage,
                .memoryDomain = MemoryBudgetDomain::Geometry}).transform([&](auto rhiValue) { geometry = std::move(rhiValue); })) {
            return RhiTestResult::fail("Released reservation did not restore allocation headroom");
        }
        const auto beforeMove = device.memoryBudget();
        const size_t domain = size_t(MemoryBudgetDomain::Geometry);
        std::shared_ptr<Buffer> retained = std::move(geometry);
        if (device.memoryBudget().domains[domain].allocationBytes != beforeMove.domains[domain].allocationBytes) {
            return RhiTestResult::fail("Retained old resource was prematurely removed from accounting");
        }
        retained.reset();
        if (device.memoryBudget().domains[domain].allocationBytes != initial.domains[domain].allocationBytes) {
            return RhiTestResult::fail("Resource destruction did not release domain accounting");
        }
        // Move assignment must release the overwritten Vulkan allocation too.
        std::unique_ptr<Buffer> a, b;
        const BufferDesc buffer{.size = kMiB, .usage = BufferUsageBits::Storage, .memoryDomain = MemoryBudgetDomain::Geometry};
        if (!device.createBuffer(buffer).transform([&](auto rhiValue) { a = std::move(rhiValue); }) || !device.createBuffer(buffer).transform([&](auto rhiValue) { b = std::move(rhiValue); })) {
            return RhiTestResult::fail("Move-assignment fixture allocation failed");
        }
        *a = std::move(*b);
        a.reset(); b.reset();
        std::unique_ptr<Texture> x, y;
        const TextureDesc texture{.usage = TextureUsageBits::Sampled, .format = Format::Rgba8Unorm,
            .width = 64, .height = 64, .memoryDomain = MemoryBudgetDomain::MaterialTextures};
        if (!device.createTexture(texture).transform([&](auto rhiValue) { x = std::move(rhiValue); }) || !device.createTexture(texture).transform([&](auto rhiValue) { y = std::move(rhiValue); })) {
            return RhiTestResult::fail("Texture move fixture allocation failed");
        }
        *x = std::move(*y);
        x.reset(); y.reset();
        MemoryBudgetReservation impossible;
        if (!hasError(device.reserveMemoryBudget(UINT64_MAX).transform([&](auto rhiValue) { impossible = std::move(rhiValue); }), Error::OutOfMemory) || impossible) {
            return RhiTestResult::fail("Oversized reservation was accepted");
        }
        const auto final = device.memoryBudget();
        for (size_t i = 0; i < final.domains.size(); ++i) {
            if (final.domains[i].allocationBytes != initial.domains[i].allocationBytes ||
                final.domains[i].allocationCount != initial.domains[i].allocationCount) {
                return RhiTestResult::fail("Failed or moved allocation leaked in domain ledger");
            }
        }
        if (final.reservedBytes != initial.reservedBytes || final.deniedAllocations < initial.deniedAllocations + 3) {
            return RhiTestResult::fail("Failed admission leaked reservation or omitted denial counter");
        }
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream(context.outputDirectory / "memory-budget.json") << nlohmann::json{
            {"driverBudget", final.driverBudget}, {"heapCount", final.heaps.size()},
            {"deniedAllocations", final.deniedAllocations - initial.deniedAllocations},
            {"reservationReleased", true}, {"resourceLedgerRestored", true}}.dump(2);
        return RhiTestResult::pass("Shared geometry/texture admission, reservations, retained resources and move/destruction accounting");
    }
};
METALLIC_REGISTER_RHI_TEST(MemoryBudgetTest);
} // namespace
} // namespace metallic::tests
