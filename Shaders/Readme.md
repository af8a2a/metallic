Metal raytraced shadows still use the native Metal shader at `Shaders/Raytracing/raytraced_shadow.metal`.

Vulkan raytraced shadows use the Slang shader at `Shaders/Raytracing/raytraced_shadow.slang`.

This split is intentional because the current local Slang toolchain cannot compile the compute + ray query shadow path to Metal.
