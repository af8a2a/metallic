#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "Certify.hpp"
#include "Contract.hpp"
#include "Export.hpp"
#include "Model.hpp"
#include "Report.hpp"
#include "Sweep.hpp"

namespace
{

// The pilot contract, embedded so --mode self-test never depends on the
// working directory. Keep in sync with contracts/AttenuationExpSmallTauV1.yaml;
// the self-test verifies file and embedded copy agree when both are present.
const char* kEmbeddedContractYaml = R"yaml(# Numeric validation contract: polynomial approximation of exp(-tau) on the
# small optical depth range, the phase-1 pilot of the shader math harness.
id: attenuation_exp_small_tau_v1

reference:
  expression: "exp(-tau)"
  interpretation: exact_real_at_binary32_input

inputs:
  tau:
    format: binary32
    domain: ["0", "1/4"]
    finite_only: true

candidate:
  form: horner_degree_4
  coefficient_bits:
    - "0x3f800000"
    - "0xbf800000"
    - "0x3f000000"
    - "0xbe2aaaab"
    - "0x3d2aaaab"
  ideal_coefficients: ["1", "-1", "1/2", "-1/6", "1/24"]
  postprocess: clamp_0_1

proof_model:
  operations: binary32_fma
  rounding: nearest_even
  subnormal_inputs: preserve
  reassociation: forbidden

requirements:
  max_absolute_error: "8.5e-6"
  output_range: ["0", "1"]
  endpoint_at_zero: "1"

outside_domain:
  action: reference_fallback
)yaml";

const char* kDefaultContractPath = "shader-math-harness/contracts/AttenuationExpSmallTauV1.yaml";
const char* kDefaultOutDirectory = "shader-math-harness-output";

struct Options
{
    bool selfTest = false;
    std::string contractPath = kDefaultContractPath;
    std::string outDirectory = kDefaultOutDirectory;
    std::uint32_t threads = 0;
    bool sampleOnly = false;
    std::uint64_t sampleCount = 1'000'000;
    std::uint64_t maxExhaustiveInputs = 0; // 0 = unlimited
    std::uint32_t worstK = 32;
    bool runSollya = false;
    bool runGappa = false;
    bool quiet = false;
};

void PrintUsage()
{
    std::printf(
        "MetallicShaderMathHarness - certified-error validation for shader numerics\n"
        "\n"
        "Usage:\n"
        "  MetallicShaderMathHarness [--mode validate|self-test] [options]\n"
        "\n"
        "Options:\n"
        "  --contract <path>            contract YAML (default: %s)\n"
        "  --out <dir>                  artifact directory (default: %s)\n"
        "  --threads <n>                sweep worker threads (default: hardware)\n"
        "  --sample <n>                 sampled sweep with n inputs instead of exhaustive\n"
        "  --max-exhaustive-inputs <n>  downgrades to sampling above n inputs (0=unlimited)\n"
        "  --worst-k <n>                worst inputs tracked and re-verified (default 32)\n"
        "  --run-sollya                 invoke Sollya on the generated script if installed\n"
        "  --run-gappa                  invoke Gappa on the generated script if installed\n"
        "  --quiet                      suppress sweep progress output\n",
        kDefaultContractPath, kDefaultOutDirectory);
}

std::optional<Options> ParseArguments(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        auto nextValue = [&]() -> std::optional<std::string>
        {
            if (i + 1 >= argc)
            {
                std::printf("missing value for %s\n", argument.c_str());
                return std::nullopt;
            }
            return std::string(argv[++i]);
        };
        if (argument == "--help" || argument == "-h")
        {
            PrintUsage();
            return std::nullopt;
        }
        if (argument == "--mode")
        {
            std::optional<std::string> value = nextValue();
            if (!value)
            {
                return std::nullopt;
            }
            if (*value == "self-test")
            {
                options.selfTest = true;
            }
            else if (*value != "validate")
            {
                std::printf("unknown mode '%s'\n", value->c_str());
                return std::nullopt;
            }
            continue;
        }
        if (argument == "--self-test")
        {
            options.selfTest = true;
            continue;
        }
        if (argument == "--contract")
        {
            std::optional<std::string> value = nextValue();
            if (!value)
            {
                return std::nullopt;
            }
            options.contractPath = *value;
            continue;
        }
        if (argument == "--out")
        {
            std::optional<std::string> value = nextValue();
            if (!value)
            {
                return std::nullopt;
            }
            options.outDirectory = *value;
            continue;
        }
        if (argument == "--threads")
        {
            std::optional<std::string> value = nextValue();
            if (!value)
            {
                return std::nullopt;
            }
            options.threads = static_cast<std::uint32_t>(std::strtoul(value->c_str(), nullptr, 10));
            continue;
        }
        if (argument == "--sample")
        {
            std::optional<std::string> value = nextValue();
            if (!value)
            {
                return std::nullopt;
            }
            options.sampleOnly = true;
            options.sampleCount = std::strtoull(value->c_str(), nullptr, 10);
            continue;
        }
        if (argument == "--max-exhaustive-inputs")
        {
            std::optional<std::string> value = nextValue();
            if (!value)
            {
                return std::nullopt;
            }
            options.maxExhaustiveInputs = std::strtoull(value->c_str(), nullptr, 10);
            continue;
        }
        if (argument == "--worst-k")
        {
            std::optional<std::string> value = nextValue();
            if (!value)
            {
                return std::nullopt;
            }
            options.worstK = static_cast<std::uint32_t>(std::strtoul(value->c_str(), nullptr, 10));
            continue;
        }
        if (argument == "--run-sollya")
        {
            options.runSollya = true;
            continue;
        }
        if (argument == "--run-gappa")
        {
            options.runGappa = true;
            continue;
        }
        if (argument == "--quiet")
        {
            options.quiet = true;
            continue;
        }
        std::printf("unknown argument '%s' (try --help)\n", argument.c_str());
        return std::nullopt;
    }
    return options;
}

std::string CaptureCommand(const std::string& command)
{
#if defined(_MSC_VER)
    std::FILE* pipe = _popen(command.c_str(), "r");
#else
    std::FILE* pipe = popen(command.c_str(), "r");
#endif
    if (pipe == nullptr)
    {
        return std::string();
    }
    std::ostringstream output;
    char buffer[1024];
    while (std::fgets(buffer, static_cast<int>(sizeof(buffer)), pipe) != nullptr)
    {
        output << buffer;
    }
#if defined(_MSC_VER)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    std::string text = output.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
    {
        text.pop_back();
    }
    return text;
}

std::string EnvironmentDescription()
{
    std::ostringstream text;
#if defined(_MSC_VER)
    text << "MSVC " << _MSC_VER;
#elif defined(__clang__)
    text << "Clang " << __clang_major__;
#elif defined(__GNUC__)
    text << "GCC " << __GNUC__;
#else
    text << "unknown compiler";
#endif
#if defined(__AVX2__)
    text << ", AVX2/FMA codegen";
#endif
    text << ", sizeof(double)=" << sizeof(double);
    return text.str();
}

std::string Timestamp()
{
    const std::time_t now =
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
#if defined(_MSC_VER)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local);
    return std::string(buffer);
}

int RunValidate(const Options& options)
{
    const auto startTime = std::chrono::steady_clock::now();

    std::string error;
    std::optional<metallic::shadermath::Contract> contract =
        metallic::shadermath::LoadContractFile(options.contractPath, error);
    if (!contract)
    {
        std::printf("failed to load contract: %s\n", error.c_str());
        return 1;
    }
    std::optional<metallic::shadermath::CandidateModel> model =
        metallic::shadermath::MakeModel(*contract, error);
    if (!model)
    {
        std::printf("failed to build candidate model: %s\n", error.c_str());
        return 1;
    }

    std::printf("certifying %s ...\n", contract->id.c_str());
    metallic::shadermath::CertificationReport certification =
        metallic::shadermath::RunCertification(*contract, *model);

    metallic::shadermath::SweepConfig sweepConfig;
    sweepConfig.tryExhaustive = !options.sampleOnly;
    sweepConfig.maxInputs = options.sampleOnly ? options.sampleCount : options.maxExhaustiveInputs;
    sweepConfig.threads = options.threads;
    sweepConfig.worstK = options.worstK;
    sweepConfig.quiet = options.quiet;
    metallic::shadermath::SweepResults sweep =
        metallic::shadermath::RunSweep(*model, *contract, sweepConfig);

    metallic::shadermath::ExportOptions exportOptions;
    exportOptions.attemptSollya = options.runSollya;
    exportOptions.attemptGappa = options.runGappa;
    metallic::shadermath::ExportArtifacts exports = metallic::shadermath::WriteExports(
        *contract, *model, sweep, options.outDirectory, exportOptions);

    metallic::shadermath::ValidationRun run;
    run.contract = &*contract;
    run.model = &*model;
    run.certification = &certification;
    run.sweep = &sweep;
    run.exports = &exports;
    run.checks = metallic::shadermath::RunConsistencyChecks(certification, sweep);
    run.environment = EnvironmentDescription() + ", " + Timestamp();
    run.gitRevision = CaptureCommand("git rev-parse --short HEAD 2>nul");
    const auto endTime = std::chrono::steady_clock::now();
    run.totalSeconds = std::chrono::duration<double>(endTime - startTime).count();

    namespace fs = std::filesystem;
    fs::create_directories(options.outDirectory);
    const std::string reportPath =
        (fs::path(options.outDirectory) / "report.json").string();
    std::ofstream report(reportPath, std::ios::binary);
    report << metallic::shadermath::BuildReportJson(run);
    report.close();

    metallic::shadermath::PrintSummary(run);
    std::printf("\nreport: %s\n", reportPath.c_str());

    const bool proved = certification.totalErrorVerdict ==
            metallic::shadermath::EvidenceLevel::ProvedUnderModel &&
        certification.outputRangeVerdict == metallic::shadermath::EvidenceLevel::ProvedUnderModel &&
        certification.endpointVerdict == metallic::shadermath::EvidenceLevel::ProvedUnderModel &&
        run.checks.AllPassed();
    return proved ? 0 : 1;
}

int RunSelfTest()
{
    using namespace metallic::shadermath;
    int failures = 0;
    int checks = 0;
    auto check = [&](const char* name, bool condition)
    {
        ++checks;
        std::printf("[%s] %s\n", condition ? "OK" : "FAIL", name);
        if (!condition)
        {
            ++failures;
        }
    };

    std::string error;
    std::optional<Contract> contract = LoadContractText(kEmbeddedContractYaml, "<embedded>", error);
    check("embedded contract parses", contract.has_value());
    if (!contract)
    {
        std::printf("error: %s\n", error.c_str());
        return 1;
    }

    {
        std::string parseError;
        const std::string withTypo = std::string(kEmbeddedContractYaml) + "\nnot_a_key: 1\n";
        const std::optional<Contract> rejected = LoadContractText(withTypo, "<typo>", parseError);
        check("unknown contract keys are rejected", !rejected.has_value() && !parseError.empty());
    }

    std::optional<CandidateModel> model = MakeModel(*contract, error);
    check("candidate model builds", model.has_value());
    if (!model)
    {
        return 1;
    }
    bool bitsRoundTrip = true;
    for (std::size_t i = 0; i < model->coefficients.size(); ++i)
    {
        bitsRoundTrip = bitsRoundTrip &&
            BitsFromFloat(model->coefficients[i]) == contract->coefficientBits[i];
    }
    check("coefficient bits round-trip exactly", bitsRoundTrip);
    check("endpoint tau=0 evaluates to exactly 1.0f",
        BitsFromFloat(EvaluateModelF32(*model, 0.0f)) == 0x3f800000u);

    const Interval expQuarter = ExpNegativeSmall(IntervalExact(0.25));
    check("rigorous exp(-1/4) contains double exp(-1/4)",
        expQuarter.lo <= std::exp(-0.25) && std::exp(-0.25) <= expQuarter.hi);
    check("rigorous exp(-1/4) width below 1e-13", expQuarter.hi - expQuarter.lo < 1e-13);
    const Interval expZero = ExpNegativeSmall(IntervalExact(0.0));
    check("rigorous exp(0) contains 1", expZero.lo <= 1.0 && 1.0 <= expZero.hi);

    const CertificationReport cert = RunCertification(*contract, *model);

    // Values cross-checked against hand analysis: true sup(P(c)-e^-x) =
    // 1595/2048 - e^{-1/4} = 7.8107e-6, analytic Taylor bound 1/122880.
    check("approximation hull completed", cert.approximationError.completed);
    check("approximation sup in [7.810e-6, 7.85e-6]",
        cert.approximationError.hull.hi >= 7.810e-6 && cert.approximationError.hull.hi <= 7.85e-6);
    check("approximation sup below analytic Taylor bound 1/122880",
        cert.approximationError.hull.hi <= 8.138021e-6);

    // True sup |P(c~)-P(c)| = 15/206158430208 = 7.2766e-11 (attained as a
    // negative difference); triangle bound from the contract document is
    // 17/206158430208 = 8.2468e-11.
    const double coefficientSup = Sup(cert.coefficientError.hull);
    check("coefficient hull completed", cert.coefficientError.completed);
    check("coefficient sup in [7.2e-11, 7.45e-11]",
        coefficientSup >= 7.2e-11 && coefficientSup <= 7.45e-11);
    check("coefficient sup below triangle bound 17/206158430208",
        coefficientSup <= 8.2469e-11);

    check("candidate real hull completed", cert.candidateRealError.completed);
    check("candidate real sup in [7.79e-6, 7.86e-6]",
        cert.candidateRealError.hull.hi >= 7.79e-6 && cert.candidateRealError.hull.hi <= 7.86e-6);
    check("triangle inequality: candidateReal <= approx + coeff",
        cert.candidateRealError.hull.hi <=
            cert.approximationError.hull.hi + cert.coefficientError.hull.hi + 1e-15);

    // gamma_4 = 1/(2^22-1) = 1/4194303; gamma_4 * sum |c~_k| (1/4)^k = 3.0613e-7.
    check("gamma bound in [3.0600e-7, 3.0620e-7]",
        cert.gammaBound.hi >= 3.0600e-7 && cert.gammaBound.hi <= 3.0620e-7);
    check("gamma bound below documented 3.062e-7", cert.gammaBound.hi <= 3.062e-7);
    check("no subnormal intermediates certified", cert.gammaNoUnderflow);

    check("total bound in [8.05e-6, 8.25e-6]",
        cert.totalErrorBound.hi >= 8.05e-6 && cert.totalErrorBound.hi <= 8.25e-6);
    check("total bound meets 8.5e-6 requirement",
        cert.totalErrorVerdict == EvidenceLevel::ProvedUnderModel);

    check("real polynomial range contains exact 1595/2048 and 1",
        cert.realPolynomialRange.lo <= 0.77880859375 && cert.realPolynomialRange.hi >= 1.0);
    check("real polynomial certified monotone decreasing",
        cert.realPolynomialMonotoneDecreasing);
    check("binary32 model range inside [0, 1]",
        cert.modelOutputRange.lo >= 0.0f && cert.modelOutputRange.hi <= 1.0f);
    check("binary32 model range lower end near 0.7788086",
        cert.modelOutputRange.lo >= 0.778807f && cert.modelOutputRange.lo <= 0.778809f);
    check("endpoint verdict proved", cert.endpointVerdict == EvidenceLevel::ProvedUnderModel);

    SweepConfig sweepConfig;
    sweepConfig.tryExhaustive = false;
    sweepConfig.maxInputs = 200'000;
    sweepConfig.worstK = 16;
    sweepConfig.quiet = true;
    const SweepResults sweep = RunSweep(*model, *contract, sweepConfig);
    check("sampled sweep ran in sampled mode", !sweep.exhaustive && sweep.evaluatedInputs > 0);
    check("observed max error below certified total",
        sweep.maxObservedError <= cert.totalErrorBound.hi + 1e-12);
    check("observed max error is in the expected magnitude band",
        sweep.maxObservedError >= 7.0e-6 && sweep.maxObservedError <= cert.totalErrorBound.hi);
    check("observed eval error below gamma bound",
        sweep.maxObservedEvalError <= cert.gammaBound.hi + 1e-12);
    check("observed output range inside certified range",
        sweep.observedMinY >= cert.modelOutputRange.lo &&
            sweep.observedMaxY <= cert.modelOutputRange.hi);
    check("worst entry certified interval contains observed",
        !sweep.worst.empty() && sweep.worst.front().certifiedError.lo - 1e-12 <=
            sweep.worst.front().observedError &&
        sweep.worst.front().observedError <= sweep.worst.front().certifiedError.hi + 1e-12);

    const std::string manifestA = BuildManifestEntriesText(*contract, *model, sweep, 2000, 16);
    const std::string manifestB = BuildManifestEntriesText(*contract, *model, sweep, 2000, 16);
    check("manifest generation is deterministic", manifestA == manifestB);

    std::printf("\nself-test: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    std::optional<Options> options = ParseArguments(argc, argv);
    if (!options)
    {
        return 2;
    }
    return options->selfTest ? RunSelfTest() : RunValidate(*options);
}
