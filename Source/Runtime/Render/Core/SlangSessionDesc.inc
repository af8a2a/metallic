    targetDesc_ = {};
    targetDesc_.format = SLANG_SPIRV;
    const char* profileName = desc.profileName != nullptr ? desc.profileName : kDefaultSlangProfileName;
    targetDesc_.profile = globalSession ? globalSession->findProfile(profileName) : SLANG_PROFILE_UNKNOWN;
    if (globalSession && targetDesc_.profile == SLANG_PROFILE_UNKNOWN) {
        log = "Unknown Slang target profile: ";
        log += profileName;
        log += '\n';
        return false;
    }

    normalizedSearchPaths_.clear();
    normalizedSearchPaths_.reserve(5u + desc.additionalSearchPaths.size());
    normalizedSearchPaths_.push_back(normalizedAbsolutePath(desc.searchPath));
    for (uint32_t index = 0; index < desc.additionalSearchPaths.size(); ++index) {
        normalizedSearchPaths_.push_back(
            normalizedAbsolutePath(desc.additionalSearchPaths[index]));
    }
    // Program roots and SDK paths retain precedence. Subsystem modules have
    // stable names, also when the entry point lives under tests or an SDK root.
    const auto addModuleRoot = [&](const std::filesystem::path& root) {
        for (const char* directory : {"Modules", "Interop"}) {
            const auto path = normalizedAbsolutePath(root / directory);
            if (std::find(normalizedSearchPaths_.begin(), normalizedSearchPaths_.end(), path) ==
                normalizedSearchPaths_.end()) {
                normalizedSearchPaths_.push_back(path);
            }
        }
    };
    addModuleRoot(desc.searchPath);
    addModuleRoot(PROJECT_SOURCE_DIR "/Shaders");
    searchPathStorage_.clear();
    searchPathStorage_.reserve(normalizedSearchPaths_.size());
    for (const std::filesystem::path& searchPath : normalizedSearchPaths_) {
        searchPathStorage_.push_back(searchPath.string());
    }
    searchPaths_.clear();
    searchPaths_.reserve(searchPathStorage_.size());
    for (const std::string& searchPath : searchPathStorage_) {
        searchPaths_.push_back(searchPath.c_str());
    }
    compilerOptions_.clear();
    compilerOptions_.reserve(desc.capabilities.size() + desc.macroDefines.size() + 8u);
    compilerOptions_.push_back(slang::CompilerOptionEntry{
        .name = slang::CompilerOptionName::EmitSpirvDirectly,
        .value = slang::CompilerOptionValue{
            .kind = slang::CompilerOptionValueKind::Int,
            .intValue0 = 1,
        },
    });
    if (cacheOptions.reportDetailedCompilerPerf) {
        // Reporting options do not alter generated code or its cache identity.
        for (const auto option : {slang::CompilerOptionName::ReportDetailedPerfBenchmark,
                slang::CompilerOptionName::ReportDownstreamTime}) {
            compilerOptions_.push_back(slang::CompilerOptionEntry{
                .name = option,
                .value = {.kind = slang::CompilerOptionValueKind::Int, .intValue0 = 1},
            });
        }
    }
    compilerOptions_.push_back(slang::CompilerOptionEntry{
        .name = slang::CompilerOptionName::Optimization,
        .value = slang::CompilerOptionValue{
            .kind = slang::CompilerOptionValueKind::Int,
            .intValue0 = static_cast<int32_t>(optimizationLevel),
        },
    });
    if (*effectivePostprocess == SlangShaderPostprocessMode::SlangDeadBranches) {
        // DownstreamArgs encodes one argument per line. Keep both passes in one
        // entry so the pinned SDK forwards the complete list in order.
        compilerOptions_.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::DownstreamArgs,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::String,
                .stringValue0 = "spirv-opt",
                .stringValue1 = "--eliminate-dead-branches\n--simplify-instructions",
            },
        });
    }
    if (native) {
        // TO-REMOVE(VVL payload-size): retain literal ArrayStride until VVL correctly
        // sizes task/mesh payloads with unrelated opaque-size spec expressions.
        // Revert to SPIRVUnifiedDescriptorHeapStride only after the raw native
        // task/mesh regression passes; then remove device ABI cache inputs and
        // warmup property query together. See Documentation/NativeDescriptorHeapStrideWorkaround.md.
        for (const auto& [option, stride] : {
                std::pair{slang::CompilerOptionName::SPIRVResourceHeapStride, resolvedDesc.descriptorHeapStrides.resource},
                std::pair{slang::CompilerOptionName::SPIRVSamplerHeapStride, resolvedDesc.descriptorHeapStrides.sampler}}) {
            compilerOptions_.push_back(slang::CompilerOptionEntry{
                .name = option,
                .value = slang::CompilerOptionValue{
                    .kind = slang::CompilerOptionValueKind::Int,
                    .intValue0 = static_cast<int32_t>(stride),
                },
            });
        }
        compilerOptions_.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::Capability,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::String,
                .stringValue0 = "spvDescriptorHeapEXT",
            },
        });
    }
    if (debugMode != SlangShaderDebugMode::Disabled) {
        compilerOptions_.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::DebugInformation,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::Int,
                // Nsight's high-level source and flame graph need embedded source
                // and NonSemantic function/call-site records. Minimal (-g1) only
                // emits legacy paths/lines and cannot provide that correlation.
                .intValue0 = static_cast<int32_t>(SLANG_DEBUG_INFO_LEVEL_STANDARD),
            },
        });
    }
    for (uint32_t capabilityIndex = 0;
         !desc.capabilities.empty() && capabilityIndex < desc.capabilities.size();
         ++capabilityIndex) {
        const char* capability = desc.capabilities[capabilityIndex];
        if (capability == nullptr || capability[0] == '\0') {
            continue;
        }
        compilerOptions_.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::Capability,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::String,
                .stringValue0 = capability,
            },
        });
    }
    for (uint32_t macroIndex = 0;
         !desc.macroDefines.empty() && macroIndex < desc.macroDefines.size();
         ++macroIndex) {
        const SlangMacroDefine& macro = desc.macroDefines[macroIndex];
        if (macro.name == nullptr || macro.name[0] == '\0') {
            continue;
        }
        compilerOptions_.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::MacroDefine,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::String,
                .stringValue0 = macro.name,
                .stringValue1 = macro.value != nullptr ? macro.value : "1",
            },
        });
    }

    compilerOptions_.push_back(slang::CompilerOptionEntry{
        .name = slang::CompilerOptionName::MacroDefine,
        .value = {.kind = slang::CompilerOptionValueKind::String,
            .stringValue0 = "METALLIC_WORKING_SPACE_ACESCG",
            .stringValue1 = sceneWorkingColorSpace() == SceneWorkingColorSpace::ACEScg ? "1" : "0"},
    });

    // Sessions cache imported modules. Keep a fresh session per request so a
    // source edit or SDK macro permutation cannot reuse stale module IR.
    sessionDesc_ = {};
    sessionDesc_.targets = &targetDesc_;
    sessionDesc_.targetCount = 1;
    sessionDesc_.searchPaths = searchPaths_.data();
    sessionDesc_.searchPathCount = static_cast<SlangInt>(searchPaths_.size());
    sessionDesc_.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    sessionDesc_.compilerOptionEntries = compilerOptions_.empty() ? nullptr : compilerOptions_.data();
    sessionDesc_.compilerOptionEntryCount = compilerOptions_.size();

    sessionDesc_.fileSystem = fileSystem;
