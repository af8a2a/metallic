# Device-generated commands

`DeviceDesc::enableDeviceGeneratedCommands` defaults to true and enables
`VK_EXT_device_generated_commands` when supported. This is an optional capability;
check `Device::capabilities().deviceGeneratedCommands` before use. The independent
`VK_KHR_device_address_commands` backend requirement still applies.
`dynamicGeneratedPipelineLayout` is enabled only if the device supports it.

The Vulkan API is in `VulkanGeneratedCommands.h`. `GeneratedCommands` owns an
indirect commands layout, an optional execution set, and its preprocess buffer.
It takes native Vulkan token descriptions to preserve the extension's draw,
dispatch, mesh, count, push-constant and descriptor-heap push-data token formats.
Token legality, enabled shader stages, pipeline/layout compatibility, and native
handle ownership follow the Vulkan specification. Check driver limits with
`queryGeneratedCommandsProperties` before choosing a layout.

## Usage

1. Create the initial pipeline or shader object. For execution-set membership,
   set `indirectBindable = true` on `ComputePipelineDesc`, `GraphicsPipelineDesc`
   or `GraphicsShaderObjectProgramDesc`. Unsupported stage combinations return
   `Unsupported`. This flag also participates in the pipeline state hash.
2. Obtain native pipeline/layout or shader handles with `nativePipeline` or
   `nativeShaders`. Build a `GeneratedCommandsDesc` containing the token layout,
   execution-set description, maximum sequence count, and maximum draw count.
   Without an execution-set token, supply a fixed `pipeline` or `shaders` list.
3. Call `initialize`. Preprocess storage uses `PREPROCESS_BUFFER` and device-address
   usage. Allocation respects the intersection of the buffer and DGC memory-type
   masks, required alignment, dedicated-allocation requirements, and zero-size
   preprocess requirements.
4. Populate command records in an RHI buffer with `BufferUsageBits::Indirect`.
   Add `Storage` if shaders generate the records. The optional count buffer also
   needs `Indirect` usage; its GPU value must not exceed the configured maximum.
   Argument offsets are multiples of four. The command range contains one full
   `indirectStride` per maximum sequence, including any trailing record padding.
5. Bind the initial pipeline/shaders, descriptors, push data, and other state not
   supplied by tokens. Call `execute` with `isPreprocessed = false` for automatic
   preprocessing. Execute graphics commands inside rendering and compute commands
   outside rendering. Rebind affected RHI state afterwards; cached state is cleared.

For explicit preprocessing, create the layout with
`VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT`. Call `preprocess`
outside rendering, passing a recording command buffer containing the execution
state; it may be the preprocessing command buffer itself. On the same queue,
call `preprocessBarrier` before `execute(..., true)`. Use the same arguments and
unchanged input data for preprocessing and execution. GPU producers must be
synchronized to `COMMAND_PREPROCESS` / `COMMAND_PREPROCESS_READ` for explicit
preprocessing, or `DRAW_INDIRECT` / `INDIRECT_COMMAND_READ` for implicit processing.
The helper's barrier covers preprocess writes to execution reads only.

`updatePipelines` and `updateShaders` update execution-set slots. After updates,
`prepare` must requery memory requirements and allocate fresh scratch; execution
is rejected until this succeeds. Do this before recording dependent commands.
Pipelines in one set must have compatible layouts and shader interfaces. Slots
referenced by GPU records must have been initialized.

## Lifetime and synchronization

The device outlives the helper. Referenced pipelines, shaders, pipeline layouts,
argument/count buffers, and the helper itself must remain alive until submitted
work finishes. Initialization/reset, updates, `prepare`, and destruction require
that prior users are idle. They do not implicitly wait for the GPU.

Each helper owns a single exclusive preprocess allocation. Use a separate helper
per frame in flight and queue family, or synchronize reuse on the same family.
Cross-queue execution additionally needs semaphores and appropriate ownership
transfers for caller-owned buffers; `preprocessBarrier` alone is insufficient.
Use native Vulkan commands for synchronization not represented by RHI barriers.

## Verification

Build `Metallic` and `MetallicRhiTests`, then run:

```powershell
build-dev/tests/MetallicRhiTests.exe --rhi-validation --gtest_filter=*DeviceGeneratedCommands*:*device_generated_commands*
```

CPU tests cover invalid objects, PSO hash separation, and probe shader compilation.
The GPU readback test covers fixed state, execution-set pipeline selection, push
constants, a nonzero stream offset, GPU sequence count, execution-set updates,
automatic/explicit preprocessing, and rejection of a misaligned stream. It skips
when the required device capabilities are unavailable.

See the [Khronos extension specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_device_generated_commands.html)
and [generated-command execution rules](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdExecuteGeneratedCommandsEXT.html).
