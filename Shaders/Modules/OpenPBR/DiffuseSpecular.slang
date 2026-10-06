#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_diffuse_specular.h. Modified by Metallic, 2026.
namespace Metallic.OpenPBR
{
/***************************************************************************
 * Copyright 2026 Adobe
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

public struct OpenPBRDiffuseSpecular
{
    public float3 diffuse;
    public float3 specular;
};

public OpenPBRDiffuseSpecular openPBRMakeDiffuseSpecular(const float3 diffuse, const float3 specular)
{
    OpenPBRDiffuseSpecular result;
    result.diffuse = diffuse;
    result.specular = specular;
    return result;
}

public OpenPBRDiffuseSpecular openPBRMakeZeroDiffuseSpecular()
{
    return openPBRMakeDiffuseSpecular(float3(0.0f), float3(0.0f));
}

public OpenPBRDiffuseSpecular openPBRMakeOneDiffuseSpecular()
{
    return openPBRMakeDiffuseSpecular(float3(1.0f), float3(1.0f));
}

public OpenPBRDiffuseSpecular openPBRMakeDiffuseSpecularFromDiffuse(const float3 diffuse)
{
    return openPBRMakeDiffuseSpecular(diffuse, float3(0.0f));
}

public OpenPBRDiffuseSpecular openPBRMakeDiffuseSpecularFromSpecular(const float3 specular)
{
    return openPBRMakeDiffuseSpecular(float3(0.0f), specular);
}

public bool openPBRIsNonzeroDiffuseSpecular(const OpenPBRDiffuseSpecular diffuse_specular)
{
    return any(((diffuse_specular.diffuse) != (float3(0.0f)))) || any(((diffuse_specular.specular) != (float3(0.0f))));
}

public OpenPBRDiffuseSpecular openPBRScaleDiffuseSpecular(const OpenPBRDiffuseSpecular diffuse_specular, const float3 scale)
{
    return openPBRMakeDiffuseSpecular(diffuse_specular.diffuse * scale, diffuse_specular.specular * scale);
}

public OpenPBRDiffuseSpecular openPBRScaleDiffuseSpecular(const OpenPBRDiffuseSpecular diffuse_specular, const float scale)
{
    return openPBRMakeDiffuseSpecular(diffuse_specular.diffuse * scale, diffuse_specular.specular * scale);
}

public float3 openPBRExtractDiffuseFromDiffuseSpecular(const OpenPBRDiffuseSpecular diffuse_specular)
{
    return diffuse_specular.diffuse;
}

public float3 openPBRExtractSpecularFromDiffuseSpecular(const OpenPBRDiffuseSpecular diffuse_specular)
{
    return diffuse_specular.specular;
}

public void openPBRSetDiffuseInDiffuseSpecular(inout OpenPBRDiffuseSpecular diffuse_specular, const float3 diffuse)
{
    diffuse_specular.diffuse = diffuse;
}

public void openPBRSetSpecularInDiffuseSpecular(inout OpenPBRDiffuseSpecular diffuse_specular, const float3 specular)
{
    diffuse_specular.specular = specular;
}

public float3 openPBRGetSumOfDiffuseSpecular(const OpenPBRDiffuseSpecular diffuse_specular)
{
    return diffuse_specular.diffuse + diffuse_specular.specular;
}

public OpenPBRDiffuseSpecular openPBRAddDiffuseSpecular(const OpenPBRDiffuseSpecular a, const OpenPBRDiffuseSpecular b)
{
    return openPBRMakeDiffuseSpecular(a.diffuse + b.diffuse, a.specular + b.specular);
}
} // namespace Metallic.OpenPBR
