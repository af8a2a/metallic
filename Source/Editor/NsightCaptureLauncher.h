#pragma once

#include <SDL3/SDL_error.h>
#include <SDL3/SDL_process.h>

#include <filesystem>
#include <string>

namespace metallic {

inline bool openNsightCapture(const std::filesystem::path& installationRoot,
    const std::filesystem::path& capturePath, std::string& error)
{
    error.clear();
    const std::u8string executable = (installationRoot /
        "host/windows-desktop-nomad-x64/ngfx-ui.exe").u8string();
    const std::u8string capture = capturePath.u8string();
    // File associations may still reference a removed Nsight version after an
    // upgrade. Launch the capture's installation directly with separate args.
    const char* args[] = {reinterpret_cast<const char*>(executable.c_str()),
        reinterpret_cast<const char*>(capture.c_str()), nullptr};
    SDL_Process* process = SDL_CreateProcess(args, false);
    if (process == nullptr) {
        error = SDL_GetError();
        return false;
    }
    SDL_DestroyProcess(process);
    return true;
}

} // namespace metallic
