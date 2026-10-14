    if (desc.moduleName == nullptr || desc.entryPointName == nullptr || desc.searchPath == nullptr ||
        (desc.additionalSearchPaths.size() > UINT32_MAX || desc.capabilities.size() > UINT32_MAX ||
         desc.macroDefines.size() > UINT32_MAX)) {
        return makeError(Error::InvalidArgument);
    }
    for (uint32_t index = 0; index < desc.additionalSearchPaths.size(); ++index) {
        if (desc.additionalSearchPaths[index] == nullptr ||
            desc.additionalSearchPaths[index][0] == '\0') {
            return makeError(Error::InvalidArgument);
        }
    }

    for (const auto& macro : desc.macroDefines) {
        if (macro.name != nullptr && std::string_view(macro.name) == "METALLIC_WORKING_SPACE_ACESCG") {
            log = "Working color space is process-wide; per-shader override is forbidden";
            return makeError(Error::InvalidArgument);
        }
    }
    const bool native = nativeDescriptorHeapEnabled(desc);
    SlangShaderDesc resolvedDesc = desc;
    resolvedDesc.descriptorHeapMode = native ? SlangDescriptorHeapMode::Native : SlangDescriptorHeapMode::Mapped;
    resolvedDesc.descriptorHeapStrides = {};
    if (native) {
        auto strides = desc.descriptorHeapStrides;
        if (!strides.resource && !strides.sampler) {
            strides = slangDescriptorHeapShaderStrides();
        }
        if (strides.resource < 8 || !strides.sampler ||
            strides.resource > INT32_MAX || strides.sampler > INT32_MAX) {
            log = "Native descriptor-heap compilation requires explicit resource/sampler strides "
                "or an initialized Vulkan device. Both strides must fit a positive Slang int32 option.\n";
            return makeError(Error::InvalidArgument);
        }
        resolvedDesc.descriptorHeapStrides = strides;
    }
    const SlangShaderDebugMode debugMode = slangShaderDebugMode();
    const auto effectiveOptimization = resolveOptimizationLevel(desc, debugMode);
    if (!effectiveOptimization) {
        log = "Invalid Slang optimization level; METALLIC_SLANG_OPTIMIZATION and explicit requests "
            "must select default, O0, O1, O2 or O3.\n";
        return makeError(Error::InvalidArgument);
    }
    const SlangOptimizationLevel optimizationLevel = compilerOptimizationLevel(*effectiveOptimization);
    const auto effectivePostprocess = resolvePostprocessMode(desc);
    if (!effectivePostprocess) {
        log = "Invalid SPIR-V postprocess mode; METALLIC_SPIRV_POSTPROCESS and explicit requests "
            "must select default, none, dead-branches or slang-dead-branches.\n";
        return makeError(Error::InvalidArgument);
    }
    if (*effectivePostprocess == SlangShaderPostprocessMode::DeadBranches ||
        *effectivePostprocess == SlangShaderPostprocessMode::SlangDeadBranches) {
        if (native) {
            log = "SPIR-V dead-branch cleanup currently supports only mapped descriptors; "
                "native descriptor-heap optimization is unsupported.\n";
            return makeError(Error::InvalidArgument);
        }
#if !METALLIC_HAS_SPIRV_TOOLS
        log = *effectivePostprocess == SlangShaderPostprocessMode::SlangDeadBranches
            ? "Slang downstream SPIR-V cleanup requires a build with the SPIRV-Tools validator.\n"
            : "SPIR-V dead-branch cleanup requires a build with SPIRV-Tools-opt.\n";
        return makeError(Error::InvalidArgument);
#endif
    }
    const auto& target = vulkanSpirvTarget();
    const uint64_t requestHash = shaderRequestHash(
        resolvedDesc, debugMode, optimizationLevel, target, *effectivePostprocess);
