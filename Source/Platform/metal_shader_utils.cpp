#include "metal_shader_utils.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace {

MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

MTL::VertexDescriptor* metalVertexDescriptor(void* handle) {
    return static_cast<MTL::VertexDescriptor*>(handle);
}

MTL::PixelFormat metalPixelFormat(RhiFormat format) {
    switch (format) {
    case RhiFormat::R8Unorm: return MTL::PixelFormatR8Unorm;
    case RhiFormat::R16Float: return MTL::PixelFormatR16Float;
    case RhiFormat::R32Float: return MTL::PixelFormatR32Float;
    case RhiFormat::R32Uint: return MTL::PixelFormatR32Uint;
    case RhiFormat::RG8Unorm: return MTL::PixelFormatRG8Unorm;
    case RhiFormat::RG16Float: return MTL::PixelFormatRG16Float;
    case RhiFormat::RG32Float: return MTL::PixelFormatRG32Float;
    case RhiFormat::RGBA8Unorm: return MTL::PixelFormatRGBA8Unorm;
    case RhiFormat::RGBA8Srgb: return MTL::PixelFormatRGBA8Unorm_sRGB;
    case RhiFormat::RGBA16Float: return MTL::PixelFormatRGBA16Float;
    case RhiFormat::RGBA32Float: return MTL::PixelFormatRGBA32Float;
    case RhiFormat::BGRA8Unorm: return MTL::PixelFormatBGRA8Unorm;
    case RhiFormat::D32Float: return MTL::PixelFormatDepth32Float;
    case RhiFormat::D16Unorm: return MTL::PixelFormatDepth16Unorm;
    case RhiFormat::Undefined:
    default: return MTL::PixelFormatInvalid;
    }
}

MTL::VertexFormat metalVertexFormat(RhiVertexFormat format) {
    switch (format) {
    case RhiVertexFormat::Float2: return MTL::VertexFormatFloat2;
    case RhiVertexFormat::Float4: return MTL::VertexFormatFloat4;
    case RhiVertexFormat::Float3:
    default:
        return MTL::VertexFormatFloat3;
    }
}

std::string metalErrorMessage(NS::Error* error, const char* fallback) {
    if (error && error->localizedDescription()) {
        return error->localizedDescription()->utf8String();
    }
    return fallback ? fallback : "Unknown Metal error";
}

MTL::Library* metalLibrary(void* handle) {
    return static_cast<MTL::Library*>(handle);
}

MTL::LanguageVersion metalLanguageVersion(uint32_t version) {
    switch (version) {
    case 31:
        return MTL::LanguageVersion3_1;
    default:
        return MTL::LanguageVersion1_0;
    }
}

MTL::Library* metalCreateLibraryImpl(void* deviceHandle,
                                     const std::string& source,
                                     const MetalShaderLibraryDesc& desc,
                                     std::string& errorMessage) {
    auto* device = metalDevice(deviceHandle);
    if (!device) {
        errorMessage = "Missing Metal device";
        return nullptr;
    }

    NS::Error* error = nullptr;
    auto* compileOptions = MTL::CompileOptions::alloc()->init();
    if (desc.languageVersion != 0) {
        compileOptions->setLanguageVersion(metalLanguageVersion(desc.languageVersion));
    }
    auto* library = device->newLibrary(
        NS::String::string(source.c_str(), NS::UTF8StringEncoding),
        compileOptions,
        &error);
    compileOptions->release();

    if (!library) {
        errorMessage = metalErrorMessage(error, "Failed to create Metal shader library");
    }

    return library;
}

} // namespace

void* metalCreateLibraryFromSource(void* deviceHandle,
                                   const std::string& source,
                                   const MetalShaderLibraryDesc& desc,
                                   std::string& errorMessage) {
    return metalCreateLibraryImpl(deviceHandle, source, desc, errorMessage);
}

void* metalCreateVertexDescriptor() {
    return MTL::VertexDescriptor::alloc()->init();
}

void metalVertexDescriptorSetAttribute(void* vertexDescriptorHandle,
                                       uint32_t attributeIndex,
                                       RhiVertexFormat format,
                                       uint32_t offset,
                                       uint32_t bufferIndex) {
    auto* descriptor = metalVertexDescriptor(vertexDescriptorHandle);
    if (!descriptor) {
        return;
    }

    auto* attribute = descriptor->attributes()->object(attributeIndex);
    attribute->setFormat(metalVertexFormat(format));
    attribute->setOffset(offset);
    attribute->setBufferIndex(bufferIndex);
}

void metalVertexDescriptorSetLayout(void* vertexDescriptorHandle,
                                    uint32_t bufferIndex,
                                    uint32_t stride) {
    auto* descriptor = metalVertexDescriptor(vertexDescriptorHandle);
    if (!descriptor) {
        return;
    }

    auto* layout = descriptor->layouts()->object(bufferIndex);
    layout->setStride(stride);
    layout->setStepFunction(MTL::VertexStepFunctionPerVertex);
}

void* metalCreateRenderPipelineFromSource(void* deviceHandle,
                                          const std::string& source,
                                          const MetalRenderPipelineDesc& desc,
                                          std::string& errorMessage) {
    auto* device = metalDevice(deviceHandle);
    if (!device) {
        errorMessage = "Missing Metal device";
        return nullptr;
    }

    auto* library = metalCreateLibraryImpl(deviceHandle, source, {}, errorMessage);
    if (!library) {
        return nullptr;
    }

    const auto colorFormat = metalPixelFormat(desc.colorFormat);
    const auto depthFormat = metalPixelFormat(desc.depthFormat);

    void* pipeline = nullptr;
    if (desc.meshEntry) {
        auto* meshFunction = library->newFunction(NS::String::string(desc.meshEntry, NS::UTF8StringEncoding));
        auto* fragmentFunction = desc.fragmentEntry
            ? library->newFunction(NS::String::string(desc.fragmentEntry, NS::UTF8StringEncoding))
            : nullptr;

        if (!meshFunction || (desc.fragmentEntry && !fragmentFunction)) {
            errorMessage = "Failed to resolve mesh pipeline entry points";
            if (meshFunction) meshFunction->release();
            if (fragmentFunction) fragmentFunction->release();
            library->release();
            return nullptr;
        }

        auto* pipelineDesc = MTL::MeshRenderPipelineDescriptor::alloc()->init();
        pipelineDesc->setMeshFunction(meshFunction);
        pipelineDesc->setFragmentFunction(fragmentFunction);
        pipelineDesc->colorAttachments()->object(0)->setPixelFormat(colorFormat);
        if (depthFormat != MTL::PixelFormatInvalid) {
            pipelineDesc->setDepthAttachmentPixelFormat(depthFormat);
        }

        NS::Error* error = nullptr;
        MTL::RenderPipelineReflection* reflection = nullptr;
        pipeline = device->newRenderPipelineState(
            pipelineDesc,
            MTL::PipelineOptionNone,
            &reflection,
            &error);

        if (reflection) {
            reflection->release();
        }
        pipelineDesc->release();
        meshFunction->release();
        if (fragmentFunction) fragmentFunction->release();

        if (!pipeline) {
            errorMessage = metalErrorMessage(error, "Failed to create Metal mesh render pipeline");
        }
    } else {
        auto* vertexFunction = desc.vertexEntry
            ? library->newFunction(NS::String::string(desc.vertexEntry, NS::UTF8StringEncoding))
            : nullptr;
        auto* fragmentFunction = desc.fragmentEntry
            ? library->newFunction(NS::String::string(desc.fragmentEntry, NS::UTF8StringEncoding))
            : nullptr;

        if ((desc.vertexEntry && !vertexFunction) || (desc.fragmentEntry && !fragmentFunction)) {
            errorMessage = "Failed to resolve render pipeline entry points";
            if (vertexFunction) vertexFunction->release();
            if (fragmentFunction) fragmentFunction->release();
            library->release();
            return nullptr;
        }

        auto* pipelineDesc = MTL::RenderPipelineDescriptor::alloc()->init();
        pipelineDesc->setVertexFunction(vertexFunction);
        pipelineDesc->setFragmentFunction(fragmentFunction);
        pipelineDesc->colorAttachments()->object(0)->setPixelFormat(colorFormat);
        if (depthFormat != MTL::PixelFormatInvalid) {
            pipelineDesc->setDepthAttachmentPixelFormat(depthFormat);
        }
        if (desc.vertexDescriptorHandle) {
            pipelineDesc->setVertexDescriptor(metalVertexDescriptor(desc.vertexDescriptorHandle));
        }

        NS::Error* error = nullptr;
        pipeline = device->newRenderPipelineState(pipelineDesc, &error);

        pipelineDesc->release();
        if (vertexFunction) vertexFunction->release();
        if (fragmentFunction) fragmentFunction->release();

        if (!pipeline) {
            errorMessage = metalErrorMessage(error, "Failed to create Metal render pipeline");
        }
    }

    library->release();
    return pipeline;
}

void* metalCreateComputePipelineFromLibrary(void* deviceHandle,
                                            void* libraryHandle,
                                            const char* entryPoint,
                                            std::string& errorMessage) {
    auto* device = metalDevice(deviceHandle);
    auto* library = metalLibrary(libraryHandle);
    if (!device) {
        errorMessage = "Missing Metal device";
        return nullptr;
    }
    if (!library) {
        errorMessage = "Missing Metal shader library";
        return nullptr;
    }

    auto* function = library->newFunction(NS::String::string(entryPoint, NS::UTF8StringEncoding));
    if (!function) {
        errorMessage = "Failed to resolve compute pipeline entry point";
        return nullptr;
    }

    NS::Error* error = nullptr;
    void* pipeline = device->newComputePipelineState(function, &error);
    function->release();

    if (!pipeline) {
        errorMessage = metalErrorMessage(error, "Failed to create Metal compute pipeline");
    }

    return pipeline;
}

void* metalCreateComputePipelineFromSource(void* deviceHandle,
                                           const std::string& source,
                                           const char* entryPoint,
                                           std::string& errorMessage) {
    auto* device = metalDevice(deviceHandle);
    if (!device) {
        errorMessage = "Missing Metal device";
        return nullptr;
    }

    auto* library = metalCreateLibraryImpl(deviceHandle, source, {}, errorMessage);
    if (!library) {
        return nullptr;
    }
    void* pipeline = metalCreateComputePipelineFromLibrary(deviceHandle, library, entryPoint, errorMessage);
    library->release();

    return pipeline;
}
