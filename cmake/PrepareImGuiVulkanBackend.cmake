# Apply the pinned ImGui backend's multi-viewport synchronization fixes to a
# build-local copy. Keep the submodule pristine and fail on upstream drift so
# an ImGui upgrade cannot silently drop or misapply these workarounds.
# Present semaphores must follow acquired images, not the acquire semaphore ring:
# https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html
function(metallic_imgui_replace_once variable before after)
    string(FIND "${${variable}}" "${before}" first)
    string(FIND "${${variable}}" "${before}" last REVERSE)
    if(first LESS 0 OR NOT first EQUAL last)
        message(FATAL_ERROR "ImGui Vulkan backend changed; review the multi-viewport patch: ${before}")
    endif()
    string(REPLACE "${before}" "${after}" patched "${${variable}}")
    set(${variable} "${patched}" PARENT_SCOPE)
endfunction()

function(metallic_prepare_imgui_vulkan_backend source destination)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
    file(READ "${source}" backend)
    string(REPLACE "\r\n" "\n" backend "${backend}")

    # Swapchain images may only be transitioned after acquisition. The helper
    # currently transitions every new image before its first acquisition.
    string(FIND "${backend}" "    // FIXME: to submit the command buffer, we need a queue." begin)
    string(FIND "${backend}" "\nvoid ImGui_ImplVulkanH_DestroyWindow(" end)
    if(begin LESS 0 OR end LESS begin)
        message(FATAL_ERROR "ImGui Vulkan resize helper changed; review its image initialization patch")
    endif()
    math(EXPR length "${end} - ${begin}")
    string(SUBSTRING "${backend}" ${begin} ${length} initialization)
    metallic_imgui_replace_once(backend "${initialization}" "}\n")
    metallic_imgui_replace_once(backend
        "barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;"
        "barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; // Discard after acquisition; the viewport is cleared.")

    # Acquire semaphores retain their existing ring. Rendering and presentation
    # use a separate semaphore indexed by the image returned by acquisition.
    metallic_imgui_replace_once(backend
        "info.pSignalSemaphores = &fsd->RenderCompleteSemaphore;"
        "info.pSignalSemaphores = &wd->FrameSemaphores[wd->FrameIndex].RenderCompleteSemaphore;")
    metallic_imgui_replace_once(backend
        "info.pWaitSemaphores = &fsd->RenderCompleteSemaphore;"
        "info.pWaitSemaphores = &wd->FrameSemaphores[wd->FrameIndex].RenderCompleteSemaphore;")
    metallic_imgui_replace_once(backend
        "    ImGui_ImplVulkanH_FrameSemaphores* fsd = &wd->FrameSemaphores[wd->SemaphoreIndex];\n    VkPresentInfoKHR info = {};"
        "    VkPresentInfoKHR info = {};")

    get_filename_component(output_directory "${destination}" DIRECTORY)
    file(MAKE_DIRECTORY "${output_directory}")
    # Avoid recompiling the backend on unrelated CMake reconfigurations.
    if(EXISTS "${destination}")
        file(READ "${destination}" previous)
        if(previous STREQUAL backend)
            return()
        endif()
    endif()
    file(WRITE "${destination}" "${backend}")
endfunction()
