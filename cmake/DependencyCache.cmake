include_guard(GLOBAL)

get_filename_component(METALLIC_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(METALLIC_DEPENDENCY_CACHE "${METALLIC_REPOSITORY_ROOT}/.cache/dependencies" CACHE PATH
    "Shared installed dependencies; keep outside application build directories")
set(METALLIC_DEPENDENCY_ROOT "" CACHE PATH "Explicit installed Metallic dependency package")

function(metallic_dependency_identity)
    if(CMAKE_CONFIGURATION_TYPES OR NOT CMAKE_BUILD_TYPE)
        message(FATAL_ERROR "Prebuilt dependencies require a single-config generator and CMAKE_BUILD_TYPE (use the Ninja presets)")
    endif()
    find_package(Git REQUIRED)
    set(identity "Metallic dependencies v1\n")
    # Preset booleans use TRUE/FALSE while option() defaults use ON/OFF.
    # These spellings describe the same package and must produce the same key.
    if(METALLIC_ENABLE_OPENUSD)
        set(METALLIC_ENABLE_OPENUSD ON)
    else()
        set(METALLIC_ENABLE_OPENUSD OFF)
    endif()
    # Deliberately conservative: do not silently mix configurations or toolchains.
    foreach(variable IN ITEMS CMAKE_VERSION CMAKE_SYSTEM_NAME CMAKE_SYSTEM_VERSION
            CMAKE_SYSTEM_PROCESSOR CMAKE_SIZEOF_VOID_P CMAKE_C_COMPILER CMAKE_C_COMPILER_ID
            CMAKE_C_COMPILER_VERSION CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER
            CMAKE_CXX_COMPILER_ID CMAKE_CXX_COMPILER_VERSION CMAKE_CXX_COMPILER_TARGET
            CMAKE_GENERATOR CMAKE_GENERATOR_PLATFORM CMAKE_GENERATOR_TOOLSET
            CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION CMAKE_BUILD_TYPE
            CMAKE_MSVC_RUNTIME_LIBRARY CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
            CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
            CMAKE_STATIC_LINKER_FLAGS CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET
            CMAKE_OSX_SYSROOT CMAKE_SYSROOT METALLIC_ENABLE_OPENUSD)
        string(APPEND identity "${variable}=${${variable}}\n")
    endforeach()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" configuration)
    foreach(kind IN ITEMS C_FLAGS CXX_FLAGS EXE_LINKER_FLAGS SHARED_LINKER_FLAGS STATIC_LINKER_FLAGS)
        string(APPEND identity "CMAKE_${kind}_${configuration}=${CMAKE_${kind}_${configuration}}\n")
    endforeach()
    foreach(variable IN ITEMS WindowsSDKVersion VCToolsVersion)
        string(APPEND identity "${variable}=$ENV{${variable}}\n")
    endforeach()
    if(CMAKE_TOOLCHAIN_FILE)
        file(SHA256 "${CMAKE_TOOLCHAIN_FILE}" toolchain_hash)
        string(APPEND identity "toolchain=${toolchain_hash}\n")
    endif()
    foreach(recipe IN ITEMS DependencyOptions.cmake DependencyCache.cmake CompilerLauncher.cmake
            RunMsvcCompiler.cmake dependencies/CMakeLists.txt dependencies/ConfigureDependency.cmake)
        file(SHA256 "${METALLIC_REPOSITORY_ROOT}/cmake/${recipe}" recipe_hash)
        string(APPEND identity "${recipe}=${recipe_hash}\n")
    endforeach()
    set(dependencies SDL3 spdlog)
    if(METALLIC_ENABLE_OPENUSD)
        list(APPEND dependencies oneTBB OpenUSD)
    endif()
    foreach(dependency IN LISTS dependencies)
        set(source "${METALLIC_REPOSITORY_ROOT}/External/${dependency}")
        if(EXISTS "${source}/.git")
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source}" rev-parse HEAD
                OUTPUT_VARIABLE revision RESULT_VARIABLE result OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT result EQUAL 0)
                message(FATAL_ERROR "Cannot read ${dependency} revision")
            endif()
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source}" diff --binary HEAD
                OUTPUT_VARIABLE changes RESULT_VARIABLE result)
            if(NOT result EQUAL 0)
                message(FATAL_ERROR "Cannot read ${dependency} changes")
            endif()
            string(SHA256 changes_hash "${changes}")
            # Include local source additions, but not ignored build output.
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source}" ls-files --others --exclude-standard
                OUTPUT_VARIABLE untracked RESULT_VARIABLE result OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT result EQUAL 0)
                message(FATAL_ERROR "Cannot read ${dependency} local files")
            endif()
            string(REPLACE "\n" ";" untracked "${untracked}")
            foreach(path IN LISTS untracked)
                file(SHA256 "${source}/${path}" file_hash)
                string(APPEND changes_hash "\n${path}=${file_hash}")
            endforeach()
        else()
            # Consumers can use a package without initializing the heavy submodules.
            execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${METALLIC_REPOSITORY_ROOT}"
                    ls-tree HEAD "External/${dependency}"
                OUTPUT_VARIABLE entry RESULT_VARIABLE result OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT result EQUAL 0 OR NOT entry MATCHES "^160000 commit ([0-9a-f]+)")
                message(FATAL_ERROR "Cannot determine locked revision for ${dependency}")
            endif()
            set(revision "${CMAKE_MATCH_1}")
            string(SHA256 changes_hash "")
        endif()
        string(APPEND identity "${dependency}=${revision}\nchanges=${changes_hash}\n")
    endforeach()
    string(SHA256 hash "${identity}")
    set(METALLIC_DEPENDENCY_HASH "${hash}" PARENT_SCOPE)
    set(METALLIC_DEPENDENCY_IDENTITY "${identity}" PARENT_SCOPE)
    if(NOT METALLIC_DEPENDENCY_ROOT)
        set(METALLIC_DEPENDENCY_ROOT "${METALLIC_DEPENDENCY_CACHE}/${hash}" PARENT_SCOPE)
    endif()
endfunction()

function(metallic_check_dependency_package)
    set(manifest "${METALLIC_DEPENDENCY_ROOT}/metallic-dependencies.txt")
    if(NOT EXISTS "${manifest}")
        message(FATAL_ERROR "Dependency package is missing: ${METALLIC_DEPENDENCY_ROOT}\n"
            "Run cmake --preset metallic-deps-debug, then cmake --build --preset metallic-deps-debug.\n"
            "For other configurations, see Documentation/Build.md; or use METALLIC_DEPENDENCY_MODE=SOURCE.")
    endif()
    file(READ "${manifest}" actual)
    if(NOT actual STREQUAL METALLIC_DEPENDENCY_IDENTITY)
        message(FATAL_ERROR "Dependency ABI/recipe mismatch at ${METALLIC_DEPENDENCY_ROOT}. Rebuild the package with matching configuration, compiler, SDK and options.")
    endif()
endfunction()
