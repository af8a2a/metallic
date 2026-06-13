#include "Runtime/Render/slang_compiler.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <cstring>

namespace metallic::render {
namespace {

void appendDiagnostics(slang::IBlob* diagnostics, std::string& outDiagnostics)
{
    if (diagnostics == nullptr || diagnostics->getBufferPointer() == nullptr || diagnostics->getBufferSize() == 0) {
        return;
    }

    const char* text = static_cast<const char*>(diagnostics->getBufferPointer());
    size_t size = diagnostics->getBufferSize();
    if (size > 0 && text[size - 1] == '\0') {
        --size;
    }

    outDiagnostics.append(text, size);
}

} // namespace

Result compileSlangShaderToSpirv(const SlangShaderDesc& desc, ShaderCompileResult& outResult)
{
    outResult = {};

    if (desc.moduleName == nullptr || desc.entryPointName == nullptr || desc.searchPath == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    Slang::ComPtr<slang::IGlobalSession> globalSession;
    if (SLANG_FAILED(slang::createGlobalSession(globalSession.writeRef())) || globalSession == nullptr) {
        return makeError(Error::Failure);
    }

    slang::TargetDesc targetDesc{};
    targetDesc.format = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("glsl_450");

    const char* searchPaths[] = {desc.searchPath};

    slang::SessionDesc sessionDesc{};
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    sessionDesc.searchPaths = searchPaths;
    sessionDesc.searchPathCount = 1;
    sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;

    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef())) || session == nullptr) {
        return makeError(Error::Failure);
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> module(session->loadModule(desc.moduleName, diagnostics.writeRef()));
    appendDiagnostics(diagnostics, outResult.diagnostics);
    if (module == nullptr) {
        return makeError(Error::Failure);
    }

    diagnostics.setNull();
    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    if (SLANG_FAILED(module->findEntryPointByName(desc.entryPointName, entryPoint.writeRef())) || entryPoint == nullptr) {
        outResult.diagnostics += "Slang entry point not found: ";
        outResult.diagnostics += desc.entryPointName;
        outResult.diagnostics += '\n';
        return makeError(Error::Failure);
    }

    slang::IComponentType* componentTypes[] = {module, entryPoint};
    Slang::ComPtr<slang::IComponentType> program;
    if (SLANG_FAILED(session->createCompositeComponentType(componentTypes, 2, program.writeRef(), diagnostics.writeRef()))
        || program == nullptr) {
        appendDiagnostics(diagnostics, outResult.diagnostics);
        return makeError(Error::Failure);
    }
    appendDiagnostics(diagnostics, outResult.diagnostics);

    diagnostics.setNull();
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    if (SLANG_FAILED(program->link(linkedProgram.writeRef(), diagnostics.writeRef())) || linkedProgram == nullptr) {
        appendDiagnostics(diagnostics, outResult.diagnostics);
        return makeError(Error::Failure);
    }
    appendDiagnostics(diagnostics, outResult.diagnostics);

    diagnostics.setNull();
    Slang::ComPtr<slang::IBlob> shaderCode;
    if (SLANG_FAILED(linkedProgram->getEntryPointCode(0, 0, shaderCode.writeRef(), diagnostics.writeRef()))
        || shaderCode == nullptr) {
        appendDiagnostics(diagnostics, outResult.diagnostics);
        return makeError(Error::Failure);
    }
    appendDiagnostics(diagnostics, outResult.diagnostics);

    const size_t byteSize = shaderCode->getBufferSize();
    if (byteSize == 0 || (byteSize % sizeof(uint32_t)) != 0) {
        outResult.diagnostics += "Slang produced invalid SPIR-V bytecode size.\n";
        return makeError(Error::Failure);
    }

    outResult.spirv.resize(byteSize / sizeof(uint32_t));
    std::memcpy(outResult.spirv.data(), shaderCode->getBufferPointer(), byteSize);
    return {};
}

} // namespace metallic::render
