#include "BufferSequence.h"
#include "Fixtures.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <set>

namespace metallic::tests::bench {
namespace {
uint64_t randomWord(uint64_t& state)
{
    uint64_t value = (state += 0x9e3779b97f4a7c15ull);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}
}
Json generateBufferSequence(uint64_t seed, uint32_t iteration, bool injected)
{
    uint64_t state = seed ^ (uint64_t(iteration) << 32);
    Json commands = Json::array();
    for (uint32_t i = 0; i < 48; ++i) {
        const auto op = randomWord(state) % 4;
        const uint32_t buffer = randomWord(state) % 4, offset = randomWord(state) % 64;
        const uint32_t count = 1 + randomWord(state) % (64 - offset);
        Json entry{{"id", i}, {"buffer", buffer}, {"offset", offset}, {"count", count}};
        if (op == 0) {
            entry["op"] = "upload"; entry["data"] = Json::array();
            for (uint32_t n = 0; n < count; ++n) { entry["data"].push_back(uint32_t(randomWord(state))); }
        } else if (op == 1) {
            entry["op"] = "fill"; entry["value"] = uint32_t(randomWord(state));
        } else if (op == 2) {
            entry["op"] = "copy"; entry["source"] = (buffer + 1 + randomWord(state) % 3) % 4;
            entry["sourceOffset"] = randomWord(state) % (65 - count);
        } else { entry["op"] = "readback"; }
        commands.push_back(entry);
    }
    if (injected) {
        commands.insert(commands.begin() + 24, Json{{"id", 1000}, {"op", "fill"}, {"buffer", 2},
            {"offset", 3}, {"count", 7}, {"value", 0xdeadbeefu}});
    }
    return {{"schema", 1}, {"generatorVersion", 1}, {"seed", seed}, {"iteration", iteration},
        {"bufferCount", 4}, {"wordCount", 64}, {"commands", commands}};
}
void validateBufferSequence(const Json& value)
{
    const auto bounded = [](const Json& number, uint64_t limit) {
        if (!(number.is_number_unsigned() || number.is_number_integer()) ||
            (number.is_number_integer() && number.get<int64_t>() < 0) || number.get<uint64_t>() > limit) {
            throw std::runtime_error("sequence integer outside budget");
        }
        return number.get<uint32_t>();
    };
    if (bounded(value.at("schema"), 1) != 1 || bounded(value.at("generatorVersion"), 1) != 1 ||
        bounded(value.at("bufferCount"), 4) != 4 || bounded(value.at("wordCount"), 64) != 64 ||
        !value.at("commands").is_array() || value.at("commands").size() > 256) {
        throw std::runtime_error("unsupported/budget-exceeding buffer sequence");
    }
    const auto& seed = value.at("seed");
    if (!seed.is_number_unsigned() && (!seed.is_number_integer() || seed.get<int64_t>() < 0)) {
        throw std::runtime_error("sequence seed must be uint64");
    }
    bounded(value.at("iteration"), UINT32_MAX);
    std::set<uint32_t> ids;
    for (const auto& command : value.at("commands")) {
        if (!ids.insert(bounded(command.at("id"), UINT32_MAX)).second) { throw std::runtime_error("duplicate command ID"); }
        const auto buffer = bounded(command.at("buffer"), 3);
        const auto offset = bounded(command.at("offset"), 63), count = bounded(command.at("count"), 64);
        if (!count || offset + count > 64) { throw std::runtime_error("invalid sequence range"); }
        const auto op = command.at("op").get<std::string>();
        if (op == "upload") {
            if (!command.at("data").is_array() || command.at("data").size() != count) { throw std::runtime_error("upload size mismatch"); }
            for (const auto& word : command.at("data")) { bounded(word, UINT32_MAX); }
        } else if (op == "fill") { bounded(command.at("value"), UINT32_MAX); }
        else if (op == "copy") {
            if (bounded(command.at("source"), 3) == buffer || bounded(command.at("sourceOffset"), 63) + count > 64) {
                throw std::runtime_error("copy overlaps or exceeds initialized buffers");
            }
        } else if (op != "readback") { throw std::runtime_error("unknown sequence operation"); }
    }
}
} // namespace metallic::tests::bench

namespace metallic::tests {
namespace {
using namespace render;
#define SEQUENCE_REQUIRE(expression) do { const auto& checked = (expression); if (!checked) { return RhiTestResult::fail(std::string(#expression) + ": " + resultToString(checked)); } } while (false)
class BufferSequenceTest : public RhiTest {
public:
    explicit BufferSequenceTest(bool injected = false) : injected_(injected)
    { type = RhiTestType::Command; name = injected ? "buffer_sequence_injected_oracle" : "buffer_sequence"; }
    std::optional<bench::Metadata> metadata() const override
    {
        auto result = bench::gpuMetadata({"buffer.sequence.shadowModel", "buffer.sequence.replay"}, bench::Layer::Rhi,
            "core", injected_ ? "sequence-fixtures" : "property", {"sequence.json", "expected.bin", "readback.bin", "sequence-diff.json"});
        result.requirements.validation = bench::Validation::Synchronization;
        return result;
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto program = context.evidence ? bench::readJson(context.evidence->root() / "input.json").at("sequence") :
            bench::generateBufferSequence(1, 0, injected_);
        bench::validateBufferSequence(program);
        if (context.evidence) { context.evidence->json("sequence.json", program); }
        std::array<std::array<uint32_t, 64>, 4> shadow{};
        std::array<std::unique_ptr<Buffer>, 4> buffers;
        std::vector<std::unique_ptr<Buffer>> staging;
        struct Checkpoint { std::string id; std::vector<uint32_t> expected; std::unique_ptr<Buffer> buffer; };
        std::vector<Checkpoint> checkpoints;
        std::array<SyncScope, 4> previous{};
        bool inject = false;
        for (auto& buffer : buffers) {
            auto created = context.device.createBuffer({.size = 256,
                .usage = BufferUsageBits::TransferSource | BufferUsageBits::TransferDestination});
            SEQUENCE_REQUIRE(created); buffer = std::move(*created);
        }
        // All resources outlive the command drain, including exceptional exits.
        bench::GpuCommands recording(context.graphicsQueue);
        SEQUENCE_REQUIRE(recording.initialize(context.device));
        const auto transition = [&](uint32_t index, AccessBits access) -> Result<> {
            const SyncScope next{PipelineStageBits::Transfer, access};
            const BufferBarrierDesc barrier{.buffer = buffers[index].get(), .before = previous[index], .after = next};
            auto result = recording.commands->synchronize({.buffers = {&barrier, 1}});
            if (result) { previous[index] = next; }
            return result;
        };
        const auto copy = [&](Buffer& source, uint32_t sourceOffset, Buffer& target, uint32_t offset, uint32_t count) -> Result<> {
            const auto from = source.slice({uint64_t(sourceOffset) * 4, uint64_t(count) * 4});
            const auto to = target.slice({uint64_t(offset) * 4, uint64_t(count) * 4});
            if (!from) { return std::unexpected(from.error()); }
            if (!to) { return std::unexpected(to.error()); }
            return recording.commands->copyBuffer(*from, *to);
        };
        const auto upload = [&](uint32_t index, uint32_t offset, std::span<const uint32_t> data) -> Result<> {
            auto buffer = context.device.createBuffer({.size = data.size_bytes(), .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload});
            if (!buffer) { return std::unexpected(buffer.error()); }
            auto* mapped = (*buffer)->map();
            if (!mapped) { return makeError(Error::InvalidArgument); }
            std::memcpy(mapped, data.data(), data.size_bytes()); (*buffer)->flush(); (*buffer)->unmap();
            staging.push_back(std::move(*buffer));
            auto ready = transition(index, AccessBits::TransferWrite);
            if (!ready) { return ready; }
            return copy(*staging.back(), 0, *buffers[index], offset, uint32_t(data.size()));
        };
        const auto checkpoint = [&](std::string id, uint32_t index, uint32_t offset, uint32_t count) -> Result<> {
            auto buffer = context.device.createBuffer({.size = uint64_t(count) * 4, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback});
            if (!buffer) { return std::unexpected(buffer.error()); }
            checkpoints.push_back({std::move(id), {shadow[index].begin() + offset, shadow[index].begin() + offset + count}, std::move(*buffer)});
            auto ready = transition(index, AccessBits::TransferRead);
            if (!ready) { return ready; }
            return copy(*buffers[index], offset, *checkpoints.back().buffer, 0, count);
        };
        // Fixed lifetime/zero-initialization prefix keeps every deletion candidate legal.
        for (uint32_t i = 0; i < 4; ++i) { SEQUENCE_REQUIRE(upload(i, 0, shadow[i])); }
        for (const auto& command : program.at("commands")) {
            const auto index = command.at("buffer").get<uint32_t>(), offset = command.at("offset").get<uint32_t>();
            const auto count = command.at("count").get<uint32_t>();
            const auto op = command.at("op").get<std::string>();
            if (op == "upload" || op == "fill") {
                auto data = op == "upload" ? command.at("data").get<std::vector<uint32_t>>() :
                    std::vector<uint32_t>(count, command.at("value").get<uint32_t>());
                std::copy(data.begin(), data.end(), shadow[index].begin() + offset);
                SEQUENCE_REQUIRE(upload(index, offset, data));
                inject |= injected_ && op == "fill" && index == 2 && data[0] == 0xdeadbeefu;
            } else if (op == "copy") {
                const auto source = command.at("source").get<uint32_t>(), from = command.at("sourceOffset").get<uint32_t>();
                std::copy_n(shadow[source].begin() + from, count, shadow[index].begin() + offset);
                SEQUENCE_REQUIRE(transition(source, AccessBits::TransferRead));
                SEQUENCE_REQUIRE(transition(index, AccessBits::TransferWrite));
                SEQUENCE_REQUIRE(copy(*buffers[source], from, *buffers[index], offset, count));
            } else { SEQUENCE_REQUIRE(checkpoint("command/" + std::to_string(command.at("id").get<uint32_t>()), index, offset, count)); }
        }
        for (uint32_t i = 0; i < 4; ++i) { SEQUENCE_REQUIRE(checkpoint("final/" + std::to_string(i), i, 0, 64)); }
        if (inject) { checkpoints[checkpoints.size() - 2].expected[0] ^= 1u; }
        const MemoryBarrierDesc host{{PipelineStageBits::Transfer, AccessBits::TransferWrite}, {PipelineStageBits::Host, AccessBits::HostRead}};
        SEQUENCE_REQUIRE(recording.commands->synchronize({.memory = {&host, 1}}));
        SEQUENCE_REQUIRE(recording.submitAndWait());
        std::vector<uint32_t> actual, expected;
        bench::Json diff{{"schema", 1}, {"equal", true}, {"injectedOracle", injected_}, {"checkpoints", bench::Json::array()}};
        for (const auto& check : checkpoints) {
            const auto* mapped = static_cast<const uint32_t*>(check.buffer->map());
            if (!mapped) { return RhiTestResult::fail("sequence readback map failed"); }
            check.buffer->invalidate();
            std::vector<uint32_t> words(mapped, mapped + check.expected.size()); check.buffer->unmap();
            const auto mismatch = std::mismatch(words.begin(), words.end(), check.expected.begin());
            const bool equal = mismatch.first == words.end();
            diff["checkpoints"].push_back({{"id", check.id}, {"offsetWords", actual.size()}, {"words", words.size()}, {"equal", equal}});
            if (!equal && diff.at("equal").get<bool>()) {
                diff["equal"] = false;
                diff["signature"] = {{"category", "readback-mismatch"}, {"checkpoint", check.id}, {"injectedOracle", injected_}};
                diff["firstMismatchWord"] = size_t(mismatch.first - words.begin());
            }
            actual.insert(actual.end(), words.begin(), words.end());
            expected.insert(expected.end(), check.expected.begin(), check.expected.end());
        }
        if (context.evidence) {
            context.evidence->bytes("readback.bin", std::as_bytes(std::span(actual)));
            context.evidence->bytes("expected.bin", std::as_bytes(std::span(expected)));
            context.evidence->json("sequence-diff.json", diff);
        }
        return diff.at("equal").get<bool>() ? RhiTestResult::pass() : RhiTestResult::fail("buffer sequence readback mismatch");
    }
private:
    bool injected_;
};
METALLIC_REGISTER_RHI_TEST(BufferSequenceTest);
} // namespace
// Explicit fault factory, excluded from the normal registry and GPU legacy run.
std::vector<RhiTestRegistry::Factory> sequenceFaultFactories()
{
    return {[] { return std::make_unique<BufferSequenceTest>(true); }};
}
} // namespace metallic::tests
