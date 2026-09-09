# 球谐 GI Shader 接口

`Shaders/Libraries/Math/SphericalHarmonics.slang` 参考 [MJP 的 SHforHLSL](https://github.com/TheRealMJP/SHforHLSL)
封装低阶球谐的类型、投影与运算。调用采用 HLSL 2021 风格的命名空间、泛型类型和
运算符；实现使用项目现有 Slang 编译链，遵循
[Slang 泛型及运算符语法](https://docs.shader-slang.org/en/latest/coming-from-hlsl.html)。
它不是可直接交给 DXC 的 HLSL 头文件，也不依赖上游完整库。
适配部分的 MIT 声明保存在 `Shaders/Licenses/SHforHLSL.txt`。

## 类型与约定

| 类型 | 阶数与用途 |
| --- | --- |
| `SH::L1` / `SH::L1RGB` | l≤1，2 个 band，4 个标量 / RGB 系数 |
| `SH::L2` / `SH::L2RGB` | l≤2，3 个 band，9 个标量 / RGB 系数 |
| `SH::L1Of<T>` / `SH::L2Of<T>` | 指定系数值类型，例如 `half`、`half3` |
| `SH::IrradianceL2RGB` | 已经完成余弦卷积的 RGB 辐照度，用于环境数据读取 |

系数存放在 `.c[i]`，数量通过 `::kCoefficientCount` 取得，`::zero()` 创建零值。
`Coefficients<T, N>` 是上述类型共用的存储/算术实现，球谐数学接口支持 N=4 或 N=9。
方向必须归一化，系数采用现有 Metallic 与 SHforHLSL 的实数基函数顺序与符号：
`[1, y, z, x, xy, yz, (3z²−1), xz, (x²−y²)]`，每项包含相应归一化常数。
Y-up 经纬环境图只决定 texel 到方向的映射，不改变这套系数约定。

## 投影与着色

每个方向的投影需要积分权重。环境贴图使用 texel 的立体角；Monte Carlo GI
采样使用 `1 / (sampleCount * pdf)`，其中 pdf 的单位为每立体角。

```hlsl
// From a shader in Shaders/Features/<Feature>/:
#include "../../Libraries/Math/SphericalHarmonics.slang"

SH::L2RGB radianceSH = SH::L2RGB::zero();
// 在采样循环中：sampleDirection 为单位向量，samplePdf 为正。
radianceSH += SH::projectOntoL2(sampleDirection, sampleRadiance) / samplePdf;
// 采样循环结束后：
radianceSH /= float(sampleCount);

float3 irradiance = SH::calculateIrradiance(radianceSH, surfaceNormal);
float3 diffuse = max(irradiance, 0.0) * diffuseAlbedo / SH::kPi;
```

`evaluate(sh, direction)` 只重建角向函数，不做卷积、曝光或非负截断。
`calculateIrradiance(radianceSH, normal)` 先施加各 band 的 `π、2π/3、π/4`
余弦卷积，再求值；Lambertian BRDF 仍需单独乘 `albedo/π`。
常量辐射亮度 L 的结果应为辐照度 πL。

运行时环境缓冲区已经保存余弦卷积后的辐照度，应直接求值：

```hlsl
SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(environmentBuffer);
float3 irradiance = environment.evaluate(environmentLocalNormal);
```

这个类型不能直接传给 `calculateIrradiance`，以免二次卷积。
`SH::toIrradiance(radianceSH)` 可将新投影的数据转为同一类型。

## 常用运算

- `+`、`-`、`+=`、`-=`：系数集合相加或相减。
- `*`、`/`、`*=`、`/=`：标量缩放或 RGB 分量调色；支持 `weight * sh`。
- `SH::lerp(a, b, weight)`：线性插值，可用于 GI probe 混合。
- `SH::dotProduct(a, b)`：系数内积，RGB 各通道独立计算。
- `SH::projectOntoL1/L2(direction, value)`：单位方向上的标量或 RGB 投影。
- `SH::truncateToL1(sh)`、`SH::toRGB(sh)`：截断阶数、标量广播到 RGB。
- `SH::convolveWithZH(sh, bandWeights)`：传入已经归一化的每 band 卷积乘数，
  分别使用 float2 / float3；`convolveWithCosineLobe` 提供漫反射余弦核。
- `SH::rotate(sh, rotation)`：正交旋转，遵循 HLSL 行向量 `mul(direction, rotation)`。
  满足 `evaluate(rotate(sh, R), mul(normal, R)) == evaluate(sh, normal)`。

半精度类型复用同一接口，方向、旋转矩阵和标量权重仍使用 float。
使用 fp16 shader 时仍需相应设备能力；环境缓冲区保持 fp32。
此封装不添加高阶 SH、GGX 近似、遮挡求解或额外的 GI 传输算法。

## 缓冲区与现有路径

环境 ABI 保持 **9 个 float4，共 144 字节**，每项 xyz 为 RGB 系数、w 为保留值。
`loadL2RGB(buffer, offset)` / `storeL2RGB(buffer, offset, sh)` 显式处理这个边界，
offset 以 float4 元素计，写入时 w=0。不要把紧凑的 `L2RGB` 直接强转为缓冲区布局。

`EnvironmentLightingPrecompute` 通过类型化投影生成系数，保留原有并行归约与
逐系数余弦卷积。`OpenPBRDirectLighting`（参考和 VBuffer 延迟路径共用）及
`VisibilityBufferShading` 通过辐照度类型读取；环境旋转、强度与 BRDF 归一化位置保持原样。

## 验证

- `spherical_harmonics_math_and_packing`：GPU 检查 L1/L2 投影、球谐加法定理、
  算术、插值、RGB 旋转与逆旋转、余弦/ZH 卷积、常量辐照度及偏移/填充；
  同时编译 half / half3 泛型入口，fp16 不参与 GPU 数值断言。
- `photometric_gpu_units_falloff_sh`：运行实际环境预计算，检查常量和含 l=1/l=2
  方向项的 HDR 图在六个轴向上的解析辐照度。
- `visibility_buffer_deferred_openpbr`：回归参考与 VBuffer 的 OpenPBR 着色一致性。

```powershell
cmake --build cmake-build-debug-visual-studio --target MetallicRhiTests --parallel 8
$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'disabled'
.\cmake-build-debug-visual-studio\tests\MetallicRhiTests.exe --rhi-validation `
  --gtest_filter="*spherical_harmonics_math_and_packing:*photometric_gpu_units_falloff_sh:*visibility_buffer_deferred_openpbr"
```

上述 GPU 验证沿用当前驱动问题排查中的内部管线缓存禁用配置。
