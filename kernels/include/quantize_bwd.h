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

typedef enum _ScaleType
{
    SINGLE_SCALE,
    PER_WEIGHT_CHANNEL,
    PER_ACTIVATION_CHANNEL
} ScaleType;

float64 v_f32_custom_RHAZ(float64 input)
{
    int64 i32_input = v_convert_f32_to_i32_b(input, SW_RZ);
    float64 f32_i32_diff = input - v_convert_i32_to_f32_b(i32_input);

    bool256 neg_pred = from_bool64(v_f32_cmp_less_b(f32_i32_diff, 0.0, 0, to_bool64((bool256){0}), 1, 0));
    bool256 less_than_half_pred = from_bool64(v_f32_cmp_less_b(v_f32_abs_b(f32_i32_diff), 0.5, 0, to_bool64(neg_pred), 1, 0));
    bool256 round_down_pred = v_i1_xor_b(neg_pred, less_than_half_pred);

    input = v_f32_nearbyint_vb(input, SW_RU, input, to_bool64(~round_down_pred), 0);
    input = v_f32_nearbyint_vb(input, SW_RD, input, to_bool64(round_down_pred), 0);

    return input;
}

void main(tensor grad_output, tensor input, tensor input_low, tensor input_range, tensor grad_input, tensor grad_input_low, tensor grad_input_range, int levels, int level_low, int level_high)
{
    const int depth = 0;
    const int width = 1;
    const int height = 2;
    const int batch = 3;
    const int fifthDim = 4;

    float64 grad_input_low_l_acc = 0.0f;
    float64 grad_input_range_l_acc = 0.0f;

    const int5 indexSpaceStart = get_index_space_offset();
    const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

    int5 ifmCoords, lowRangeCoords = {0, 0, 0, 0, 0};
    int5 zeroCoords = {0};

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

    int scale_dim = 5;
    int scale_count = 1;
    int is_2D = 0;

    ScaleType scale_type = SINGLE_SCALE;
    for (int i = 0; i < scale_dim; i++)
    {
        scale_count *= get_dim_size(input_range, i);
    }

    if (scale_count > 1)
    {
        is_2D = (get_dim_size(input_range, 2) == 1 && get_dim_size(input_range, 3) == 1 && get_dim_size(input_range, 4) == 1) ? 1 : 0;
        if (get_dim_size(input_range, (is_2D) ? 1 : 3) > 1)
        {
            scale_type = PER_WEIGHT_CHANNEL;
        }
        else if (get_dim_size(input_range, (is_2D) ? 0 : 2) > 1)
        {
            scale_type = PER_ACTIVATION_CHANNEL;
        }
    }

    if (is_2D)
    {
        ifmCoords[height] = heightStart;
        ifmCoords[batch] = batchStart;
        ifmCoords[fifthDim] = fifthDimStart;
#pragma loop_taken
        for (int d = depthStart; d < depthEnd; d += depthStep)
        {
            ifmCoords[depth] = d;
#pragma loop_taken
#pragma unroll 4
            for (int w = widthStart; w < widthEnd; w += 1)
            {
                ifmCoords[width] = w;
                if (scale_type == PER_WEIGHT_CHANNEL)
                    lowRangeCoords[width] = w;

                float64 input_val = v_f32_ld_tnsr_b(ifmCoords, input);
                float64 input_low_val = s_f32_ld_g((__global__ float *)gen_addr(lowRangeCoords, input_low));
                float64 input_range_val = s_f32_ld_g((__global__ float *)gen_addr(lowRangeCoords, input_range));

                float64 scale = (levels - 1) / input_range_val;

                float64 output_val = v_f32_max_b(v_f32_min_b(input_val, input_low_val + input_range_val), input_low_val);
                float64 zero_point = (-input_low_val * scale);
                zero_point = v_f32_custom_RHAZ(zero_point);

                output_val -= input_low_val;
                output_val *= scale;
                output_val -= zero_point;

                output_val = v_f32_custom_RHAZ(output_val);
                output_val = output_val / scale;

                float64 grad_output_val = v_f32_ld_tnsr_b(ifmCoords, grad_output);
                float64 range_sign = (input_range_val >= 0.0f) ? 1.0f : 0.0f;

                float64 mask_hi = (input_val > (input_low_val + input_range_val)) ? 1.0f : 0.0f;
                float64 mask_lo = (input_val < input_low_val) ? 1.0f : 0.0f;
                float64 mask_in = 1 - mask_hi - mask_lo;

                float64 grad_input_val = grad_output_val * mask_in;
                v_f32_st_tnsr(ifmCoords, grad_input, grad_input_val);

                float64 grad_low_val = grad_output_val * (mask_hi + mask_lo);
                grad_input_low_l_acc += v_f32_reduce_add(grad_low_val);

                float64 err = (output_val - input_val) * v_reciprocal_f32(input_range_val * range_sign);
                float64 grad_range_val = grad_output_val * (err * mask_in + range_sign * ((float)level_low / level_high) * mask_lo + mask_hi);
                grad_input_range_l_acc += v_f32_reduce_add(grad_range_val);

                if (scale_type == PER_WEIGHT_CHANNEL)
                {
                    float64 grad_input_low_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_low);
                    grad_input_low_g_acc += grad_input_low_l_acc;
                    v_f32_st_tnsr(lowRangeCoords, grad_input_low, grad_input_low_g_acc);
                    grad_input_low_l_acc = 0.0f;

                    float64 grad_input_range_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_range);
                    grad_input_range_g_acc += grad_input_range_l_acc;
                    v_f32_st_tnsr(lowRangeCoords, grad_input_range, grad_input_range_g_acc);
                    grad_input_range_l_acc = 0.0f;
                }
            }
            if (scale_type == PER_ACTIVATION_CHANNEL)
            {
                float64 grad_input_low_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_low);
                grad_input_low_g_acc += grad_input_low_l_acc;
                v_f32_st_tnsr(lowRangeCoords, grad_input_low, grad_input_low_g_acc);
                grad_input_low_l_acc = 0.0f;

                float64 grad_input_range_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_range);
                grad_input_range_g_acc += grad_input_range_l_acc;
                v_f32_st_tnsr(lowRangeCoords, grad_input_range, grad_input_range_g_acc);
                grad_input_range_l_acc = 0.0f;
            }
        }
        if (scale_type == SINGLE_SCALE)
        {
            float64 grad_input_low_g_acc = v_f32_ld_tnsr_b(zeroCoords, grad_input_low);
            grad_input_low_g_acc += grad_input_low_l_acc;
            v_f32_st_tnsr(zeroCoords, grad_input_low, grad_input_low_g_acc);

            float64 grad_input_range_g_acc = v_f32_ld_tnsr_b(zeroCoords, grad_input_range);
            grad_input_range_g_acc += grad_input_range_l_acc;
            v_f32_st_tnsr(zeroCoords, grad_input_range, grad_input_range_g_acc);
        }
    }
    else
    {
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
                    if (scale_type == PER_WEIGHT_CHANNEL)
                    {
                        lowRangeCoords[batch] = b;
                    }
#pragma loop_taken
                    for (int h = heightStart; h < heightEnd; h += heightStep)
                    {
                        ifmCoords[height] = h;
                        if (scale_type == PER_ACTIVATION_CHANNEL)
                        {
                            lowRangeCoords[height] = h;
                        }
#pragma loop_taken
#pragma unroll 4
                        for (int w = widthStart; w < widthEnd; w += 1)
                        {
                            ifmCoords[width] = w;

                            float64 input_val = v_f32_ld_tnsr_b(ifmCoords, input);
                            float64 input_low_val = s_f32_ld_g((__global__ float *)gen_addr(lowRangeCoords, input_low));
                            float64 input_range_val = s_f32_ld_g((__global__ float *)gen_addr(lowRangeCoords, input_range));

                            float64 scale = (levels - 1) / input_range_val;

                            float64 output_val = v_f32_max_b(v_f32_min_b(input_val, input_low_val + input_range_val), input_low_val);
                            float64 zero_point = (-input_low_val * scale);
                            zero_point = v_f32_custom_RHAZ(zero_point);

                            output_val -= input_low_val;
                            output_val *= scale;
                            output_val -= zero_point;

                            output_val = v_f32_custom_RHAZ(output_val);
                            output_val = output_val / scale;

                            float64 grad_output_val = v_f32_ld_tnsr_b(ifmCoords, grad_output);
                            float64 range_sign = (input_range_val >= 0.0f) ? 1.0f : 0.0f;

                            float64 mask_hi = (input_val > (input_low_val + input_range_val)) ? 1.0f : 0.0f;
                            float64 mask_lo = (input_val < input_low_val) ? 1.0f : 0.0f;
                            float64 mask_in = 1 - mask_hi - mask_lo;

                            float64 grad_input_val = grad_output_val * mask_in;
                            v_f32_st_tnsr(ifmCoords, grad_input, grad_input_val);

                            float64 grad_low_val = grad_output_val * (mask_hi + mask_lo);
                            grad_input_low_l_acc += v_f32_reduce_add(grad_low_val);

                            float64 err = (output_val - input_val) * v_reciprocal_f32(input_range_val * range_sign);
                            float64 grad_range_val = grad_output_val * (err * mask_in + range_sign * ((float)level_low / level_high) * mask_lo + mask_hi);

                            grad_input_range_l_acc += v_f32_reduce_add(grad_range_val);
                        }
                        if (scale_type == PER_ACTIVATION_CHANNEL)
                        {
                            float64 grad_input_low_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_low);
                            grad_input_low_g_acc += grad_input_low_l_acc;
                            v_f32_st_tnsr(lowRangeCoords, grad_input_low, grad_input_low_g_acc);
                            grad_input_low_l_acc = 0.0f;

                            float64 grad_input_range_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_range);
                            grad_input_range_g_acc += grad_input_range_l_acc;
                            v_f32_st_tnsr(lowRangeCoords, grad_input_range, grad_input_range_g_acc);
                            grad_input_range_l_acc = 0.0f;
                        }
                    }
                    if (scale_type == PER_WEIGHT_CHANNEL)
                    {
                        float64 grad_input_low_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_low);
                        grad_input_low_g_acc += grad_input_low_l_acc;
                        v_f32_st_tnsr(lowRangeCoords, grad_input_low, grad_input_low_g_acc);
                        grad_input_low_l_acc = 0.0f;

                        float64 grad_input_range_g_acc = v_f32_ld_tnsr_b(lowRangeCoords, grad_input_range);
                        grad_input_range_g_acc += grad_input_range_l_acc;
                        v_f32_st_tnsr(lowRangeCoords, grad_input_range, grad_input_range_g_acc);
                        grad_input_range_l_acc = 0.0f;
                    }
                }
            }
        }
        if (scale_type == SINGLE_SCALE)
        {
            float64 grad_input_low_g_acc = v_f32_ld_tnsr_b(zeroCoords, grad_input_low);
            grad_input_low_g_acc += grad_input_low_l_acc;
            v_f32_st_tnsr(zeroCoords, grad_input_low, grad_input_low_g_acc);

            float64 grad_input_range_g_acc = v_f32_ld_tnsr_b(zeroCoords, grad_input_range);
            grad_input_range_g_acc += grad_input_range_l_acc;
            v_f32_st_tnsr(zeroCoords, grad_input_range, grad_input_range_g_acc);
        }
    }
}
