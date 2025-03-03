/**********************************************************************
Copyright (c) 2022 Habana Labs.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "printf_test.hpp"
#include "batch_norm_f32.hpp"
#include "cast_gaudi.hpp"
#include "filter_fwd_2d_bf16.hpp"
#include "softmax_bf16.hpp"
#include "leakyrelu_f32_gaudi.hpp"
#include "sparse_lengths_sum_bf16.hpp"
#include "customdiv_fwd_f32.hpp"
#include "relu6_all.hpp"
#include "matrix_mul_fwd_f32.hpp"
#include "spatial_conv_f32.hpp"
#include "sin_f32.hpp"
#include "add_f32.hpp"
#include "avg_pool_2d_f32.hpp"
#include "avg_pool_2d_f32_gaudi2.hpp"
#include "cast_f16_to_i16_gaudi2.hpp"

#include "entry_points.hpp"

extern "C"
{

gcapi::GlueCodeReturn_t GetKernelNames(_OUT_ char**         names,
                                       unsigned*            kernelCount,
                                       gcapi::DeviceId_t    deviceId)
{
    if (deviceId == gcapi::DEVICE_ID_GAUDI)
    {
        if (names != nullptr )
        {
           BatchNormF32 batchNormInstance;
           batchNormInstance.GetKernelName(names[GAUDI_KERNEL_BATCH_NORM_F32]);
           CastGaudi castInstance(CastGaudi::bf16_to_f32);
           castInstance.GetKernelName(names[GAUDI_KERNEL_CAST_BF16_F32], CastGaudi::bf16_to_f32);
           CastGaudi castInstance2(CastGaudi::f32_to_bf16);
           castInstance2.GetKernelName(names[GAUDI_KERNEL_CAST_F32_BF16], CastGaudi::f32_to_bf16);
           FilterFwd2dBF16 filterInstance;
           filterInstance.GetKernelName(names[GAUDI_KERNEL_FILTER_FWD_2D_BF16]);
           LeakyReluF32Gaudi leakyReluInstance;
           leakyReluInstance.GetKernelName(names[GAUDI_KERNEL_LEAKU_RELU_F32]);
           SoftMaxBF16 softmaxInstance;
           softmaxInstance.GetKernelNameFcd(names[GAUDI_KERNEL_SOFTMAX_FCD_BF16]);
           softmaxInstance.GetKernelNameNonFcd(names[GAUDI_KERNEL_SOFTMAX_NONFCD_BF16]);
           SparseLengthsSumBF16 sparseLengthsSumInstance;
           sparseLengthsSumInstance.GetKernelName(names[GAUDI_KERNEL_SPARSE_LEN_SUM_BF16]);
           CustomdivFwdF32 customdivFwdF32Instance;
           customdivFwdF32Instance.GetKernelName(names[GAUDI_KERNEL_CUSTOMDIV_FWD_F32]);
           Relu6All Relu6FwdF32Instance(Relu6All::relu6_fwd_f32);
           Relu6FwdF32Instance.GetKernelName(names[GAUDI_KERNEL_RELU6_FWD_F32], Relu6All::relu6_fwd_f32);
           Relu6All Relu6BwdF32Instance(Relu6All::relu6_bwd_f32);
           Relu6BwdF32Instance.GetKernelName(names[GAUDI_KERNEL_RELU6_BWD_F32], Relu6All::relu6_bwd_f32);
           Relu6All Relu6FwdBF16Instance(Relu6All::relu6_fwd_bf16);
           Relu6FwdBF16Instance.GetKernelName(names[GAUDI_KERNEL_RELU6_FWD_BF16], Relu6All::relu6_fwd_bf16);
           Relu6All Relu6BwdBF16Instance(Relu6All::relu6_bwd_bf16);
           Relu6BwdBF16Instance.GetKernelName(names[GAUDI_KERNEL_RELU6_BWD_BF16], Relu6All::relu6_bwd_bf16);
           Relu6All ReluFwdF32Instance(Relu6All::relu_fwd_f32);
           ReluFwdF32Instance.GetKernelName(names[GAUDI_KERNEL_RELU_FWD_F32], Relu6All::relu_fwd_f32);
           Relu6All ReluBwdF32Instance(Relu6All::relu_bwd_f32);
           ReluBwdF32Instance.GetKernelName(names[GAUDI_KERNEL_RELU_BWD_F32], Relu6All::relu_bwd_f32);
           Relu6All ReluFwdBF16Instance(Relu6All::relu_fwd_bf16);
           ReluFwdBF16Instance.GetKernelName(names[GAUDI_KERNEL_RELU_FWD_BF16], Relu6All::relu_fwd_bf16);
           Relu6All ReluBwdBF16Instance(Relu6All::relu_bwd_bf16);
           ReluBwdBF16Instance.GetKernelName(names[GAUDI_KERNEL_RELU_BWD_BF16], Relu6All::relu_bwd_bf16);
           MatrixMulFwdF32 MatrixMulFwdF32Instance;
           MatrixMulFwdF32Instance.GetKernelName(names[GAUDI_KERNEL_MATRIXMUL_FWD_F32]);
           SpatialConvF32 spatialConvInstance;
           spatialConvInstance.GetKernelName(names[GAUDI_KERNEL_SPATIAL_CONV_F32]);
           SinF32 sinf32Instance;
           sinf32Instance.GetKernelName(names[GAUDI_KERNEL_SIN_F32]);
           AddF32 addf32Instance;
           addf32Instance.GetKernelName(names[GAUDI_KERNEL_ADD_F32]);
           AvgPool2dF32 avgpool2dfwdf32Instance(AvgPool2dF32::fwd);
           avgpool2dfwdf32Instance.GetKernelName(names[GAUDI_KERNEL_AVG_POOL_2D_FWD_F32]);
           AvgPool2dF32 avgpool2dbwdf32Instance(AvgPool2dF32::bwd);
           avgpool2dbwdf32Instance.GetKernelName(names[GAUDI_KERNEL_AVG_POOL_2D_BWD_F32]);

        }

        if (kernelCount != nullptr)
        {
            // currently the library support 8 kernel.
            *kernelCount = GAUDI_KERNEL_MAX_EXAMPLE_KERNEL;
        }
    }
    else if (deviceId == gcapi::DEVICE_ID_GAUDI2)
    {
        if (names != nullptr )
        {
           AvgPool2dF32Gaudi2 avgpool2dfwdf32g2Instance(AvgPool2dF32Gaudi2::fwd);
           avgpool2dfwdf32g2Instance.GetKernelName(names[GAUDI2_KERNEL_AVG_POOL_2D_FWD_F32]);
           AvgPool2dF32Gaudi2 avgpool2dbwdf32g2Instance(AvgPool2dF32Gaudi2::bwd);
           avgpool2dbwdf32g2Instance.GetKernelName(names[GAUDI2_KERNEL_AVG_POOL_2D_BWD_F32]);
           Castf16toi16Gaudi2 castf16toi16g2Instance;
           castf16toi16g2Instance.GetKernelName(names[GAUDI2_KERNEL_CAST_F16_TO_I16]);

        }

        if (kernelCount != nullptr)
        {
            // currently the library support 8 kernel.
            *kernelCount = GAUDI2_KERNEL_MAX_EXAMPLE_KERNEL;
        }
    }
    else
    {
        if (kernelCount != nullptr)
        {
            // currently the library support 0 kernels.
            *kernelCount = 0;
        }
    }
    return gcapi::GLUE_SUCCESS;
}


gcapi::GlueCodeReturn_t
HabanaKernel(_IN_  gcapi::HabanaKernelParams_t* params,
             _OUT_ gcapi::HabanaKernelInstantiation_t*instance)
{
    char kernelName [gcapi::MAX_NODE_NAME];

    ///////---Gaudi---
    ///////////////////////////////
    PrintfTestKernel printfInstance;
    printfInstance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return printfInstance.GetGcDefinitions(params, instance);
    }

    BatchNormF32 batchNormInstance;
    batchNormInstance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return batchNormInstance.GetGcDefinitions(params, instance);
    }
    FilterFwd2dBF16 filterBF16Instance;
    filterBF16Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return filterBF16Instance.GetGcDefinitions(params, instance);
    }
    CastGaudi castGaudiInstancebff(CastGaudi::bf16_to_f32);
    castGaudiInstancebff.GetKernelName(kernelName, CastGaudi::bf16_to_f32);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return castGaudiInstancebff.GetGcDefinitions(params,instance);
    }
    CastGaudi castGaudiInstancefbf(CastGaudi::f32_to_bf16);
    castGaudiInstancefbf.GetKernelName(kernelName, CastGaudi::f32_to_bf16);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return castGaudiInstancefbf.GetGcDefinitions(params,instance);
    }
    LeakyReluF32Gaudi leakyReluGaudiInstance;
    leakyReluGaudiInstance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return leakyReluGaudiInstance.GetGcDefinitions(params,instance);
    }
    SoftMaxBF16 softmaxBf16Instance;
    softmaxBf16Instance.GetKernelNameFcd(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return softmaxBf16Instance.GetGcDefinitions(params,instance);
    }
    softmaxBf16Instance.GetKernelNameNonFcd(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return softmaxBf16Instance.GetGcDefinitions(params,instance);
    }
    SparseLengthsSumBF16 sparseLengthsSumBf16Instance;
    sparseLengthsSumBf16Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return sparseLengthsSumBf16Instance.GetGcDefinitions(params, instance);
    }
    CustomdivFwdF32 customdivFwdF32Instance;
    customdivFwdF32Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return customdivFwdF32Instance.GetGcDefinitions(params,instance);
    }
    Relu6All Relu6FwdF32Instance(Relu6All::relu6_fwd_f32);
    Relu6FwdF32Instance.GetKernelName(kernelName, Relu6All::relu6_fwd_f32);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return Relu6FwdF32Instance.GetGcDefinitions(params,instance);
    }

    Relu6All Relu6BwdF32Instance(Relu6All::relu6_bwd_f32);
    Relu6BwdF32Instance.GetKernelName(kernelName, Relu6All::relu6_bwd_f32);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return Relu6BwdF32Instance.GetGcDefinitions(params,instance);
    }

    Relu6All Relu6FwdBF16Instance(Relu6All::relu6_fwd_bf16);
    Relu6FwdBF16Instance.GetKernelName(kernelName, Relu6All::relu6_fwd_bf16);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return Relu6FwdBF16Instance.GetGcDefinitions(params,instance);
    }

    Relu6All Relu6BwdBF16Instance(Relu6All::relu6_bwd_bf16);
    Relu6BwdBF16Instance.GetKernelName(kernelName, Relu6All::relu6_bwd_bf16);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return Relu6BwdBF16Instance.GetGcDefinitions(params,instance);
    }

    Relu6All ReluFwdF32Instance(Relu6All::relu_fwd_f32);
    ReluFwdF32Instance.GetKernelName(kernelName, Relu6All::relu_fwd_f32);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return ReluFwdF32Instance.GetGcDefinitions(params,instance);
    }

    Relu6All ReluBwdF32Instance(Relu6All::relu_bwd_f32);
    ReluBwdF32Instance.GetKernelName(kernelName, Relu6All::relu_bwd_f32);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return ReluBwdF32Instance.GetGcDefinitions(params,instance);
    }

    Relu6All ReluFwdBF16Instance(Relu6All::relu_fwd_bf16);
    ReluFwdBF16Instance.GetKernelName(kernelName, Relu6All::relu_fwd_bf16);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return ReluFwdBF16Instance.GetGcDefinitions(params,instance);
    }

    Relu6All ReluBwdBF16Instance(Relu6All::relu_bwd_bf16);
    ReluBwdBF16Instance.GetKernelName(kernelName, Relu6All::relu_bwd_bf16);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return ReluBwdBF16Instance.GetGcDefinitions(params,instance);
    }

    MatrixMulFwdF32 MatrixMulFwdF32Instance;
    MatrixMulFwdF32Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return MatrixMulFwdF32Instance.GetGcDefinitions(params,instance);
    }

    SpatialConvF32 spatialConvInstance;
    spatialConvInstance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return spatialConvInstance.GetGcDefinitions(params, instance);
    }

    SinF32 sinf32Instance;
    sinf32Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return sinf32Instance.GetGcDefinitions(params, instance);
    }

    AddF32 addf32Instance;
    addf32Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return addf32Instance.GetGcDefinitions(params, instance);
    }

    AvgPool2dF32 avgpool2dfwdf32Instance(AvgPool2dF32::fwd);
    avgpool2dfwdf32Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return avgpool2dfwdf32Instance.GetGcDefinitions(params, instance);
    }

    AvgPool2dF32 avgpool2dbwdf32Instance(AvgPool2dF32::bwd);
    avgpool2dbwdf32Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return avgpool2dbwdf32Instance.GetGcDefinitions(params, instance);
    }

    /////// --- Gaudi2 
    ///////////////////////////////
    AvgPool2dF32Gaudi2 avgpool2dfwdf32g2Instance(AvgPool2dF32Gaudi2::fwd);
    avgpool2dfwdf32g2Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return avgpool2dfwdf32g2Instance.GetGcDefinitions(params, instance);
    }

    AvgPool2dF32Gaudi2 avgpool2dbwdf32g2Instance(AvgPool2dF32Gaudi2::bwd);
    avgpool2dbwdf32g2Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return avgpool2dbwdf32g2Instance.GetGcDefinitions(params, instance);
    }

    Castf16toi16Gaudi2 castf16toi16g2Instance;
    castf16toi16g2Instance.GetKernelName(kernelName);
    if (strcmp(params->nodeName, kernelName) == 0)
    {
        return castf16toi16g2Instance.GetGcDefinitions(params, instance);
    }

    return gcapi::GLUE_NODE_NOT_FOUND;
}

} // extern "C"
