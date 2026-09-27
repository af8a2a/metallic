#include "Evidence.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace metallic::tests::bench {

std::string fileHash(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read evidence: " + path.string()); }
    uint64_t hash = 14695981039346656037ull;
    char buffer[8192];
    while (input.read(buffer, sizeof(buffer)) || input.gcount()) {
        for (std::streamsize i = 0; i < input.gcount(); ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= 1099511628211ull;
        }
    }
    if (!input.eof()) { throw std::runtime_error("evidence read failed"); }
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << hash;
    return text.str();
}

void writeJson(const std::filesystem::path& path, const Json& value)
{
    auto temp = path;
    temp += ".tmp";
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << value.dump(2) << '\n';
    output.close();
    // Each document is finalized once in a fresh output directory.
    std::filesystem::rename(temp, path);
}

Json readJson(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("missing result: " + path.string()); }
    return Json::parse(input);
}

Evidence::Evidence(std::filesystem::path root) : root_(std::move(root))
{
    std::filesystem::create_directories(root_);
}

std::filesystem::path Evidence::resolve(const std::string& filename) const
{
    const std::filesystem::path relative(filename);
    if (relative.empty() || relative.has_parent_path() || relative.is_absolute()) {
        throw std::runtime_error("evidence filenames must be local basenames");
    }
    return root_ / relative;
}

void Evidence::json(const std::string& filename, const Json& value)
{
    writeJson(resolve(filename), value);
}

void Evidence::bytes(const std::string& filename, std::span<const std::byte> value)
{
    std::ofstream output(resolve(filename), std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(reinterpret_cast<const char*>(value.data()), std::streamsize(value.size()));
    output.close();
}

void Evidence::phase(const std::string& phase)
{
    std::ofstream output(root_ / "journal.jsonl", std::ios::binary | std::ios::app);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output << Json{{"phase", phase}}.dump() << '\n';
    output.close();
}

Json Evidence::manifest() const
{
    Json files = Json::array();
    for (const auto& entry : std::filesystem::directory_iterator(root_)) {
        const auto filename = entry.path().filename().string();
        if (!entry.is_regular_file() || filename == "stdout.log" || filename == "stderr.log" ||
            filename == "result.json" || filename.ends_with(".tmp")) { continue; }
        files.push_back({{"file", filename}, {"bytes", entry.file_size()}, {"fnv1a64", fileHash(entry.path())}});
    }
    return files;
}

} // namespace metallic::tests::bench
