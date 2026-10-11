# Streamed geometry attributes

Raster shading and ray-query shading decode positions and attributes from the
same meshlet page payloads. Ray tracing uses CLAS exclusively: the hit's cluster
ID identifies the page and cluster, and its triangle index selects the local
indices. `StreamRayGeometry.slang` resolves those indices through the live page
table and interpolates the attributes with the hit barycentrics.

| Attribute | Packed encoding |
| --- | --- |
| Shading normal | Octahedral snorm16 x 2 |
| Tangent | Octahedral snorm15 x 2, handedness and validity bits |
| UV0 | float32 x 2 |

The page format records each stream's encoding and offset; float attribute
payloads are also supported. Positions and triangle indices remain in the
streamed geometry payload. There is no separate resident ray-tracing vertex
buffer or position-fetch fallback.

CPU direction packing lives in `Source/Runtime/Scene/GeometryEncoding.h`.
`Shaders/Modules/Core/GeometryEncoding.slang` supplies the matching decoders,
and `StreamAttributeDecode.slang` selects the decoder for each page stream.
Missing normals preserve geometric-normal fallback. Tangent handedness remains
explicit, and UVs retain float32 precision for large, tiled and negative values.

The runtime instance table supplies world transforms, material bindings and
GPUScene instance IDs. Normals use the inverse-transpose transform; authored
tangents use the world transform and determinant sign. Missing tangents are
reconstructed from triangle positions and UVs. Authored normals and geometric
normals keep their orientation through tangent-space construction and normal-map
evaluation. Any face-forward operation applies only to the final shading normal.
Committed-hit world position uses `ray.origin + ray.direction * CommittedRayT()`.

The `scene_clas_attributes`, `scene_clas_attributes_authored_tangents` and
`scene_clas_attributes_native` GPU regressions exercise streamed indexed
triangles under nonuniform scale and translation, front/back hits, misses,
world-space normals, UV interpolation, authored tangent handedness and fallback
tangent reconstruction. They validate attribute results independently of the
cooked triangle ordering. CLAS relocation, TLAS updates and alpha acceptance
have separate RHI regressions.
