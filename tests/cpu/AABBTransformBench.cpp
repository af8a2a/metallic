#include "AABBTestData.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
{
    using namespace metallic::cpu;
    uint32_t count = 65536, samples = 31;
    for (int argument = 1; argument < argc; argument += 2) {
        if (argument + 1 >= argc) { return 2; }
        const std::string_view name(argv[argument]), value(argv[argument + 1]);
        uint32_t parsed = 0;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0) { return 2; }
        if (name == "--count") { count = parsed; }
        else if (name == "--samples") { samples = parsed; }
        else { return 2; }
    }
    test::AABBTestData data(count);
    std::vector<std::array<float, 18>> aos(count);
    for (uint32_t index = 0; index < count; ++index) {
        for (uint32_t element = 0; element < 18; ++element) {
            aos[index][element] = data.inputs[element][index];
        }
    }
    std::vector backends{CPUKernelBackend::Scalar, CPUKernelBackend::AutoVectorized};
    if (cpuKernelBackendAvailable(CPUKernelBackend::ISPC)) { backends.push_back(CPUKernelBackend::ISPC); }
    std::cerr << "Single caller thread; deterministic seed=0x73b01; SoA prepared before timing; "
              << "24 float streams; no task scheduler, validation layer or ISA flag on host C++.\n";
    std::cout << "phase,round,backend,count,wall_ns,checksum\n";
    for (uint64_t round = 0; round <= uint64_t(samples) + 2; ++round) {
        for (size_t step = 0; step < backends.size(); ++step) {
            const auto backend = backends[(step + round) % backends.size()];
            const auto start = std::chrono::steady_clock::now();
            const auto actual = transformAABBs(data.input(), data.output(), 0, count, backend);
            const auto finish = std::chrono::steady_clock::now();
            if (round == 1 || round == 2) { continue; }
            double checksum = 0;
            for (const auto& stream : data.outputs) {
                checksum += stream.front() + stream[count / 2] + stream.back();
            }
            std::cout << (round == 0 ? "first_call" : "warm") << ','
                      << (round == 0 ? 0 : round - 3) << ',' << cpuKernelBackendName(actual) << ',' << count << ','
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count()
                      << ',' << checksum << '\n';
        }
        const auto packStart = std::chrono::steady_clock::now();
        for (uint32_t index = 0; index < count; ++index) {
            for (uint32_t element = 0; element < 18; ++element) {
                data.inputs[element][index] = aos[index][element];
            }
        }
        const auto packEnd = std::chrono::steady_clock::now();
        if (round != 1 && round != 2) {
            double checksum = 0;
            for (const auto& stream : data.inputs) { checksum += stream[count / 2]; }
            std::cout << (round == 0 ? "first_call" : "warm") << ','
                      << (round == 0 ? 0 : round - 3) << ",aos-to-soa," << count << ','
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(packEnd - packStart).count()
                      << ',' << checksum << '\n';
        }
    }
}
