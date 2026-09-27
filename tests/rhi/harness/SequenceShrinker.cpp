#include "BufferSequence.h"
#include "Runner.h"
#include <algorithm>
#include <iostream>

namespace metallic::tests::bench {
namespace {
std::string argument(const std::filesystem::path& path)
{
    const auto value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}
Json signature(const std::filesystem::path& directory, const Json& result)
{
    if (result.at("status") != "Fail" || !result.at("executed").get<bool>()) { return nullptr; }
    // These artifacts must be part of the verified manifest even for a failed case.
    for (const auto* required : {"sequence.json", "sequence-diff.json", "readback.bin", "expected.bin"}) {
        bool found = false;
        for (const auto& file : result.at("files")) { found |= file.at("file") == required; }
        if (!found) { return nullptr; }
    }
    const auto validation = readJson(directory / "validation.json");
    if (validation.at("captureFailed").get<bool>() || validation.at("count").get<uint64_t>()) { return nullptr; }
    const auto diff = readJson(directory / "sequence-diff.json");
    if (diff.at("equal").get<bool>() || diff.at("signature").at("category") != "readback-mismatch") { return nullptr; }
    return diff.at("signature");
}
}
int shrinkBufferSequence(const std::filesystem::path& executable, const std::filesystem::path& original,
    const Json& input, const std::filesystem::path& output)
{
    const auto originalInput = readJson(original / "input.json");
    const auto expected = signature(original, verifyChild(original, originalInput, {1, false}));
    if (expected.is_null()) { throw std::runtime_error("shrink requires a verified readback mismatch without validation errors"); }
    const auto originalDevice = readJson(original / "capabilities.json");
    Evidence evidence(output);
    auto current = input.at("sequence");
    validateBufferSequence(current);
    evidence.json("original.json", current);
    Json attempts = Json::array();
    constexpr size_t budget = 64;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    const auto available = [&] { return attempts.size() + 2 <= budget && std::chrono::steady_clock::now() < deadline; };
    const auto reproduce = [&](const Json& sequence, const std::string& phase) {
        for (int repeat = 0; repeat < 2; ++repeat) {
            if (attempts.size() >= budget || std::chrono::steady_clock::now() >= deadline) { return false; }
            const auto directory = output / "attempts" / std::to_string(attempts.size());
            auto candidate = input; candidate["sequence"] = sequence;
            Evidence sample(directory); sample.json("input.json", candidate); sample.phase("scheduled");
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            const auto process = runProcess(executable, {"--tb-child", "--tb-input", argument(directory / "input.json")}, directory,
                std::min(remaining, std::chrono::milliseconds(15000)));
            const auto result = verifyChild(directory, candidate, process);
            sample.json("parent-result.json", result);
            const auto observed = signature(directory, result);
            if (result.at("status") == "Pass" || !observed.is_null()) {
                const auto actualDevice = readJson(directory / "capabilities.json");
                for (const auto* key : {"uuid", "driverVersion", "driverInfo", "apiVersion", "validationMode"}) {
                    if (actualDevice.at(key) != originalDevice.at(key)) { throw std::runtime_error("shrink device/driver/validation changed"); }
                }
            }
            const bool same = observed == expected;
            attempts.push_back({{"directory", argument(std::filesystem::relative(directory, output))},
                {"phase", phase}, {"commands", sequence.at("commands").size()}, {"status", result.at("status")},
                {"signature", observed}, {"preserved", same}});
            evidence.json("attempts.json", attempts);
            // A different failure is never accepted as a reduction. Stop on infrastructure/GPU failures.
            if (result.at("status") != "Pass" && result.at("status") != "Fail") {
                throw std::runtime_error("shrinker interrupted by " + result.at("status").get<std::string>());
            }
            if (!same) { return false; }
        }
        return true;
    };
    if (!reproduce(current, "original-confirmation")) { throw std::runtime_error("original failure did not reproduce twice"); }
    evidence.json("minimal.json", current);
    size_t chunks = 2;
    bool minimal = current.at("commands").empty();
    while (!current.at("commands").empty() && available()) {
        const auto size = current.at("commands").size();
        const auto width = (size + chunks - 1) / chunks;
        bool reduced = false;
        size_t tested = 0;
        for (size_t start = 0; start < size && available(); start += width) {
            auto candidate = current;
            auto& commands = candidate["commands"];
            commands.erase(commands.begin() + start, commands.begin() + std::min(start + width, size));
            validateBufferSequence(candidate);
            ++tested;
            if (reproduce(candidate, "candidate")) {
                current = std::move(candidate); evidence.json("minimal.json", current);
                chunks = std::max(size_t(2), chunks - 1); reduced = true; break;
            }
        }
        if (reduced) { continue; }
        if (width == 1 && tested == size) { minimal = true; break; }
        chunks = std::min(size, chunks * 2);
    }
    if (current.at("commands").empty()) { minimal = true; }
    // Every accepted program already reproduced twice; save a directly replayable input.
    auto replay = input; replay["sequence"] = current;
    Evidence(output / "minimal").json("input.json", replay);
    evidence.json("shrink.json", {{"schema", 1}, {"signature", expected}, {"originalCommands", input.at("sequence").at("commands").size()},
        {"minimalCommands", current.at("commands").size()}, {"oneDeletionMinimal", minimal}, {"budgetExhausted", !minimal},
        {"childBudget", budget}, {"wallBudgetSeconds", 120}, {"children", attempts.size()}, {"confirmationsPerAcceptedProgram", 2}});
    std::cout << "Shrunk " << input.at("sequence").at("commands").size() << " -> " << current.at("commands").size()
        << " commands; one-deletion minimal=" << minimal << "; evidence: " << output.string() << '\n';
    return 0;
}
} // namespace metallic::tests::bench
