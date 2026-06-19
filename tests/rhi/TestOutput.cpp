#include "RhiTest.h"

#include "stb/stb_image_write.h"

#include <limits>

namespace metallic::tests {

bool saveRgba8Png(
    const std::filesystem::path& outputPath,
    const uint8_t* pixels,
    uint32_t width,
    uint32_t height,
    std::string& outMessage)
{
    outMessage.clear();

    if (pixels == nullptr || width == 0 || height == 0) {
        outMessage = "saveRgba8Png received an empty image";
        return false;
    }
    if (width > static_cast<uint32_t>(std::numeric_limits<int>::max() / 4)) {
        outMessage = "saveRgba8Png image width is too large";
        return false;
    }
    if (height > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        outMessage = "saveRgba8Png image height is too large";
        return false;
    }

    std::error_code error;
    const std::filesystem::path parentPath = outputPath.parent_path();
    if (!parentPath.empty()) {
        std::filesystem::create_directories(parentPath, error);
        if (error) {
            outMessage = "failed to create output directory '" + parentPath.string() + "': " + error.message();
            return false;
        }
    }

    const std::string outputPathString = outputPath.string();
    const int strideBytes = static_cast<int>(width * 4u);
    if (stbi_write_png(
            outputPathString.c_str(),
            static_cast<int>(width),
            static_cast<int>(height),
            4,
            pixels,
            strideBytes) == 0) {
        outMessage = "failed to write PNG '" + outputPathString + "'";
        return false;
    }

    return true;
}

} // namespace metallic::tests
