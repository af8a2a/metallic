#include "Runtime/Scene/MeshletStreamGpuCodec.h"
#include <cstdio>
#include <exception>
#include <string_view>

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4 || (argc == 4 && std::string_view(argv[3]) != "--raw")) {
        std::fprintf(stderr, "MetallicMeshletTranscode input.meshstream.bin output.meshstream.bin [--raw]\n");
        return 1;
    }
    std::string reason;
    try {
        if (metallic::scene::transcodeMeshletStreamAsset(argv[1], argv[2], argc != 4, reason,
                [](uint32_t done, uint32_t total) {
                    std::printf("Transcode %u / %u pages\n", done, total); std::fflush(stdout);
                })) { return 0; }
    } catch (const std::exception& error) { reason = error.what(); }
    std::fprintf(stderr, "%s\n", reason.c_str());
    return 1;
}
