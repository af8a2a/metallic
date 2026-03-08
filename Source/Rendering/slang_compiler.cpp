#include "slang_compiler.h"

#include <slang.h>
#include <slang-com-ptr.h>
#include <spdlog/spdlog.h>
#include <cstring>
#include <regex>
#include <vector>

namespace {

std::vector<uint32_t> compileSlangComponentToSpirv(const char* shaderPath,
                                                   const char* searchPath,
                                                   const std::vector<const char*>& entryPoints,
                                                   const char* label) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef()))) {
        spdlog::error("Slang: failed to create SPIR-V session for {}", shaderPath);
        return {};
    }

    spdlog::info("Loading {} shader: {} (search path: {})",
                 label, shaderPath, searchPath ? searchPath : "<cwd>");

    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics) {
            spdlog::error("Slang load error: {}",
                          static_cast<const char*>(diagnostics->getBufferPointer()));
        }
        return {};
    }

    std::vector<slang::IComponentType*> components;
    components.reserve(entryPoints.size() + 1);
    components.push_back(module);

    std::vector<Slang::ComPtr<slang::IEntryPoint>> resolvedEntries;
    resolvedEntries.reserve(entryPoints.size());

    for (const char* entryPointName : entryPoints) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        module->findEntryPointByName(entryPointName, entryPoint.writeRef());
        if (!entryPoint) {
            spdlog::error("Slang: entry point '{}' not found in {}", entryPointName, shaderPath);
            return {};
        }
        components.push_back(entryPoint);
        resolvedEntries.push_back(entryPoint);
    }

    Slang::ComPtr<slang::IComponentType> program;
    if (SLANG_FAILED(session->createCompositeComponentType(
            components.data(), components.size(), program.writeRef(), diagnostics.writeRef()))) {
        if (diagnostics) {
            spdlog::error("Slang program creation error: {}",
                          static_cast<const char*>(diagnostics->getBufferPointer()));
        }
        return {};
    }

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    if (SLANG_FAILED(program->link(linkedProgram.writeRef(), diagnostics.writeRef()))) {
        if (diagnostics) {
            spdlog::error("Slang link error: {}",
                          static_cast<const char*>(diagnostics->getBufferPointer()));
        }
        return {};
    }

    Slang::ComPtr<slang::IBlob> spirvCode;
    if (SLANG_FAILED(linkedProgram->getTargetCode(0, spirvCode.writeRef(), diagnostics.writeRef())) || !spirvCode) {
        if (diagnostics) {
            spdlog::error("Slang SPIR-V compile error: {}",
                          static_cast<const char*>(diagnostics->getBufferPointer()));
        }
        return {};
    }

    const size_t byteSize = spirvCode->getBufferSize();
    if (byteSize == 0 || (byteSize % sizeof(uint32_t)) != 0) {
        spdlog::error("Slang returned invalid SPIR-V blob size {} for {}", byteSize, shaderPath);
        return {};
    }

    std::vector<uint32_t> spirvWords(byteSize / sizeof(uint32_t));
    std::memcpy(spirvWords.data(), spirvCode->getBufferPointer(), byteSize);
    return spirvWords;
}

} // namespace

std::string compileSlangToMetal(const char* shaderPath, const char* searchPath) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    spdlog::info("Loading shader: {} (search path: {})", shaderPath, searchPath ? searchPath : "<cwd>");
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
        spdlog::error("Slang load error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    Slang::ComPtr<slang::IEntryPoint> vertexEntry;
    module->findEntryPointByName("vertexMain", vertexEntry.writeRef());
    Slang::ComPtr<slang::IEntryPoint> fragmentEntry;
    module->findEntryPointByName("fragmentMain", fragmentEntry.writeRef());

    std::vector<slang::IComponentType*> components = {module, vertexEntry, fragmentEntry};
    Slang::ComPtr<slang::IComponentType> program;
    session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    program->link(linkedProgram.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IBlob> metalCode;
    linkedProgram->getTargetCode(0, metalCode.writeRef(), diagnostics.writeRef());
    if (!metalCode) {
        if (diagnostics)
        spdlog::error("Slang compile error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

std::string compileSlangMeshShaderToMetal(const char* shaderPath, const char* searchPath) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    spdlog::info("Loading shader: {} (search path: {})", shaderPath, searchPath ? searchPath : "<cwd>");
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
        spdlog::error("Slang load error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    Slang::ComPtr<slang::IEntryPoint> meshEntry;
    module->findEntryPointByName("meshMain", meshEntry.writeRef());
    Slang::ComPtr<slang::IEntryPoint> fragmentEntry;
    module->findEntryPointByName("fragmentMain", fragmentEntry.writeRef());

    std::vector<slang::IComponentType*> components = {module, meshEntry, fragmentEntry};
    Slang::ComPtr<slang::IComponentType> program;
    session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    program->link(linkedProgram.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IBlob> metalCode;
    linkedProgram->getTargetCode(0, metalCode.writeRef(), diagnostics.writeRef());
    if (!metalCode) {
        if (diagnostics)
        spdlog::error("Slang mesh shader compile error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

std::string compileSlangComputeShaderToMetal(const char* shaderPath, const char* searchPath, const char* entryPoint) {
    Slang::ComPtr<slang::IGlobalSession> globalSession;
    slang::createGlobalSession(globalSession.writeRef());

    slang::SessionDesc sessionDesc = {};
    slang::TargetDesc targetDesc = {};
    targetDesc.format = SLANG_METAL;
    targetDesc.profile = globalSession->findProfile("metal");
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    if (searchPath) {
        sessionDesc.searchPaths = &searchPath;
        sessionDesc.searchPathCount = 1;
    }

    Slang::ComPtr<slang::ISession> session;
    globalSession->createSession(sessionDesc, session.writeRef());

    spdlog::info("Loading shader: {} (search path: {})", shaderPath, searchPath ? searchPath : "<cwd>");
    Slang::ComPtr<slang::IBlob> diagnostics;
    slang::IModule* module = session->loadModule(shaderPath, diagnostics.writeRef());
    if (!module) {
        if (diagnostics)
        spdlog::error("Slang load error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    Slang::ComPtr<slang::IEntryPoint> computeEntry;
    module->findEntryPointByName(entryPoint, computeEntry.writeRef());
    if (!computeEntry) {
        spdlog::error("Slang: entry point '{}' not found in {}", entryPoint, shaderPath);
        return {};
    }

    std::vector<slang::IComponentType*> components = {module, computeEntry};
    Slang::ComPtr<slang::IComponentType> program;
    session->createCompositeComponentType(
        components.data(), components.size(), program.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IComponentType> linkedProgram;
    program->link(linkedProgram.writeRef(), diagnostics.writeRef());

    Slang::ComPtr<slang::IBlob> metalCode;
    linkedProgram->getTargetCode(0, metalCode.writeRef(), diagnostics.writeRef());
    if (!metalCode) {
        if (diagnostics)
        spdlog::error("Slang compute shader compile error: {}",
                      static_cast<const char*>(diagnostics->getBufferPointer()));
        return {};
    }

    return std::string(static_cast<const char*>(metalCode->getBufferPointer()),
                       metalCode->getBufferSize());
}

std::vector<uint32_t> compileSlangToSpirv(const char* shaderPath, const char* searchPath) {
    return compileSlangComponentToSpirv(shaderPath, searchPath, {"vertexMain", "fragmentMain"}, "graphics");
}

std::vector<uint32_t> compileSlangMeshShaderToSpirv(const char* shaderPath, const char* searchPath) {
    return compileSlangComponentToSpirv(shaderPath, searchPath, {"meshMain", "fragmentMain"}, "mesh");
}

std::vector<uint32_t> compileSlangComputeShaderToSpirv(const char* shaderPath,
                                                       const char* searchPath,
                                                       const char* entryPoint) {
    return compileSlangComponentToSpirv(shaderPath, searchPath, {entryPoint}, "compute");
}

std::string patchMeshShaderMetalSource(const std::string& source) {
    std::string patched = source;

    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewNormal\w*)\s*;)"),
        "$1 [[user(NORMAL)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((float3\s+\w*viewPos\w*)\s*;)"),
        "$1 [[user(TEXCOORD)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((float2\s+\w*uv\w*)\s*;)"),
        "$1 [[user(TEXCOORD_1)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((\[\[flat\]\]\s+uint\s+\w*materialID\w*)\s*;)"),
        "$1 [[user(TEXCOORD_2)]];");

    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+))"),
        "$1 [[texture(0)]]");

    return patched;
}

std::string patchVisibilityShaderMetalSource(const std::string& source) {
    std::string patched = source;

    patched = std::regex_replace(patched,
        std::regex(R"((float2\s+\w*uv\w*)\s*;)"),
        "$1 [[user(TEXCOORD)]];");

    patched = std::regex_replace(patched,
        std::regex(R"((uint\s+\w*visibility\w*)\s*;)"),
        "$1 [[user(TEXCOORD_1)]];");
    patched = std::regex_replace(patched,
        std::regex(R"((uint\s+\w*materialID\w*)\s*;)"),
        "$1 [[user(TEXCOORD_2)]];");

    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+))"),
        "$1 [[texture(0)]]");

    return patched;
}

std::string patchComputeShaderMetalSource(const std::string& source) {
    std::string patched = source;

    patched = std::regex_replace(patched,
        std::regex(R"((array<texture2d<float,\s*access::sample>,\s*int\(\d+\)>\s+\w+)(\s*,))"),
        "$1 [[texture(3)]]$2");

    return patched;
}
