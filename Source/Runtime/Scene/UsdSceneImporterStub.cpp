#include "UsdSceneImporter.h"

#include <algorithm>
#include <cctype>

namespace metallic::scene::detail {

bool isUsdScenePath(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension == ".usd" || extension == ".usda" ||
        extension == ".usdc" || extension == ".usdz";
}

bool importUsdScene(const std::filesystem::path&, UsdImportedScene& imported)
{
    imported = {};
    imported.error = "USD import is disabled in this build. Configure with METALLIC_ENABLE_OPENUSD=ON.";
    return false;
}

} // namespace metallic::scene::detail
