/**********************************************************************
Copyright (c) 2022 Habana Labs. All rights reserved.

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
// #pragma tpc_printf(enable)

#include "kernel_config.h"

void main(tensor input, tensor input_low, tensor input_range, tensor output, int levels)
{
    const int depth = 0;
    const int width = 1;
    const int height = 2;
    const int batch = 3;
    const int fifthDim = 4;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords = {0, 0, 0, 0, 0};

    // DEPTH
    const int depthStep = VECTOR_SIZE;
    const int depthStart = indexSpaceStart[depth] * depthStep;
    const int depthEnd = indexSpaceEnd[depth] * depthStep;

    // WIDTH
    const int widthStep = 4;
    const int widthStart = indexSpaceStart[width] * widthStep;
    const int widthEnd = indexSpaceEnd[width] * widthStep;

    // HEIGHT
    const int heightStep = 1;
    const int heightStart = indexSpaceStart[height];
    const int heightEnd = indexSpaceEnd[height];

    // BATCH
    const int batchStep = 1;
    const int batchStart = indexSpaceStart[batch];
    const int batchEnd = indexSpaceEnd[batch];

    // fifthDim
    const int fifthDimStep = 1;
    const int fifthDimStart = indexSpaceStart[fifthDim];
    const int fifthDimEnd = indexSpaceEnd[fifthDim];

    // load tensors as scalars, only works for 'single_scale' mode
    float input_low_val = s_f32_ld_g((__global__ float *)gen_addr(ifmCoords, input_low));
    float input_range_val = s_f32_ld_g((__global__ float *)gen_addr(ifmCoords, input_range));

#pragma loop_taken
    for (int d = depthStart; d < depthEnd; d += depthStep)
    {
        ifmCoords[depth] = d;

#pragma loop_taken
        for (int f = fifthDimStart; f < fifthDimEnd; f += fifthDimStep)
        {
            ifmCoords[fifthDim] = f;

#pragma loop_taken
            for (int b = batchStart; b < batchEnd; b += batchStep)
            {
                ifmCoords[batch] = b;

#pragma loop_taken
                for (int h = heightStart; h < heightEnd; h += heightStep)
                {
                    ifmCoords[height] = h;

#pragma loop_taken
#pragma unroll 4
                    for (int w = widthStart; w < widthEnd; w += 1)
                    {
                        ifmCoords[width] = w;

                        float64 input_val = v_f32_ld_tnsr_b(ifmCoords, input);
                        float64 scale = (levels - 1) / input_range_val;
                        float64 output_val = v_f32_max_b(v_f32_min_b(input_val, input_low_val + input_range_val), input_low_val);
                        float64 zero_point = v_f32_nearbyint_b(-input_low_val * scale);

                        output_val -= input_low_val;
                        output_val *= scale;
                        output_val -= zero_point;
                        output_val = v_f32_nearbyint_b(output_val);
                        output_val = output_val / scale;

                        v_f32_st_tnsr(ifmCoords, output, output_val);
                    }
                }
            }
        }
    }
}
