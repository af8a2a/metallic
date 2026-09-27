#pragma once

#include <json.hpp>
#include <filesystem>
#include <span>
#include <string>

namespace metallic::tests::bench {

using Json = nlohmann::ordered_json;
std::string fileHash(const std::filesystem::path& path);
void writeJson(const std::filesystem::path& path, const Json& value);
Json readJson(const std::filesystem::path& path);

// IO failures throw to the runner, which records InfrastructureFailure.
class Evidence {
public:
    explicit Evidence(std::filesystem::path root);
    const std::filesystem::path& root() const { return root_; }
    void json(const std::string& filename, const Json& value);
    void bytes(const std::string& filename, std::span<const std::byte> value);
    void phase(const std::string& phase);
    Json manifest() const;
private:
    std::filesystem::path resolve(const std::string& filename) const;
    std::filesystem::path root_;
};

} // namespace metallic::tests::bench
