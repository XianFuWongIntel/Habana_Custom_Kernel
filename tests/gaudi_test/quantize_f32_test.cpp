/**********************************************************************
Copyright (c) 2021 Habana Labs. All rights reserved.

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

#include "quantize_f32_test.hpp"

#define DEBUG_PRINTF 0
#define LOAD_FROM_JSON 1
#define JSON_FILE "../test_binaries/quant_io.json"

#if LOAD_FROM_JSON
#include "json.hpp"
#include <fstream>
using json = nlohmann::json;
#endif

void QuantizeF32Test::quantize_f32_reference_implementation(
    const float_5DTensor &input,
    const float_5DTensor &input_low,
    const float_5DTensor &input_range,
    const int levels,
    float_5DTensor &output, Gaudi_Kernel_Name_e mode)
{
    int coords[5] = {0};
    const float_5DTensor scale;

    float input_low_val = input_low.ElementAt(coords);
    float input_range_val = input_range.ElementAt(coords);

    for (unsigned f = 0; f < input.Size(4); f += 1)
    {
        coords[4] = f;
        for (unsigned b = 0; b < input.Size(3); b += 1)
        {
            coords[3] = b;
            for (unsigned h = 0; h < input.Size(2); h += 1)
            {
                coords[2] = h;
                for (unsigned w = 0; w < input.Size(1); w += 1)
                {
                    coords[1] = w;
                    for (unsigned d = 0; d < input.Size(0); d += 1)
                    {
                        coords[0] = d;
                        if (mode == GAUDI_KERNEL_QUANTIZE_FWD_F32)
                        {
                            float input_val = input.ElementAt(coords);

                            float scale = (levels - 1) / input_range_val;
                            float min_clip = input_low_val;
                            float max_clip = input_low_val + input_range_val;
                            float clipped = (input_val < min_clip) ? min_clip : (input_val > max_clip) ? max_clip
                                                                                                       : input_val;

                            clipped -= input_low_val;
                            clipped *= scale;
                            clipped = std::round(clipped);
                            clipped = clipped / scale;
                            clipped += input_low_val;

                            output.SetElement(coords, clipped);
                        }
                        else if (mode == GAUDI_KERNEL_QUANTIZE_BWD_F32)
                        {
                            // float g = gradin.ElementAt(coords);
                            // float x = input.ElementAt(coords);
                            // float y = (x < 0.0f) ? 0 : x;
                            // x = (y > 0.0f) ? 1 : 0;
                            // x = x * g;
                            // output.SetElement(coords, x);
                        }
                    }
                }
            }
        }
    }
}

void calcScale(float_5DTensor &input, float_5DTensor &scale)
{
    auto inData = input.Data();
    auto nElem = input.ElementCount();

    const auto minVal = inData[std::min_element(inData, inData + nElem) - inData];
    const auto maxVal = inData[std::max_element(inData, inData + nElem) - inData];
    const auto meanVal = std::accumulate(inData, inData + nElem, 0) / float(nElem);

    scale.Data()[0] = (std::min(std::abs(minVal), std::abs(maxVal)) - meanVal) / float(4);
    scale.Data()[0] = std::abs(scale.Data()[0]) + 1e-6;

#if DEBUG_PRINTF
    std::cout << "Elems\t: " << input.ElementCount() << std::endl;
    std::cout << "Min\t: " << minVal << std::endl;
    std::cout << "Max\t: " << maxVal << std::endl;
    std::cout << "Mean\t: " << meanVal << std::endl;
#endif
}

void setInputLowAndRange(float_5DTensor &scale, const int level_low, const int level_high, float_5DTensor &input_low, float_5DTensor &input_range)
{
    for (int element = 0; element < input_low.ElementCount(); element++)
    {
        input_low.Data()[element] = scale.Data()[element] * (level_low / (float)level_high);
        input_range.Data()[element] = scale.Data()[element] - input_low.Data()[element];
    }
}

int QuantizeF32Test::runTest(Gaudi_Kernel_Name_e NameofKernel)
{
#if LOAD_FROM_JSON
    std::ifstream test_file(JSON_FILE, std::ifstream::binary);
    json test_bin;
    test_file >> test_bin;
    auto test_content = test_bin["ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT"]["Forward"];

    std::vector<std::vector<std::vector<std::vector<float>>>> test_input = test_content["i"]["input_"];
    std::vector<float> test_input_low = test_content["i"]["input_low"];
    std::vector<float> test_input_range = test_content["i"]["input_range"];
    int test_levels = test_content["i"]["levels"].get<int>();
    std::vector<std::vector<std::vector<std::vector<float>>>> test_output_ref = test_content["o"];

    // Note: Treat input as NHWC, even thought it's supposed be to NCHW ...
    // e.g.
    // N = 64     N = 64
    // C = 3      H = 3
    // H = 7  --> W = 7
    // W = 7      C = 7
    const uint height = test_input[0].size();
    const uint width = test_input[0][0].size();
    const uint depth = test_input[0][0][0].size();
    const uint batch = test_input.size();
    ;
    const uint fifthdim = 1;
#else
    const int height = 6;
    const int width = 5;
    const int depth = 4;
    const int batch = 3;
    const int fifthdim = 2;
#endif
    unsigned int fmInitializer[] = {depth, width, height, batch, fifthdim};
    unsigned int scaleinitializer[] = {1, 1, 1, 1, 1};

    unsigned kernelCount;
    gcapi::GlueCodeReturn_t result;
    char **kernelNames = nullptr;

    const int level_low = -128;
    const int level_high = 127;

    QuantizeF32::QuantizeParam param;
    param.levels = std::abs(level_low) + std::abs(level_high) + 1;

    float_5DTensor input(fmInitializer);
    input.InitRand(-10.0f, 10.0f);
    float_5DTensor input_low(scaleinitializer);
    float_5DTensor input_range(scaleinitializer);
    float_5DTensor scale(scaleinitializer);

    // Calculate scale for 'single_scale' mode
    calcScale(input, scale);
    // Set 'input_low' and 'input_range' based on 'scale', 'level_low' and 'level_high'
    setInputLowAndRange(scale, level_low, level_high, input_low, input_range);

    float_5DTensor output(fmInitializer);
    float_5DTensor output_ref(fmInitializer);

#if LOAD_FROM_JSON
    input.loadData(test_input);
    input_low.loadData(test_input_low);
    input_range.loadData(test_input_range);
    param.levels = test_levels;
#endif
#if DEBUG_PRINTF
    std::cout << "input" << std::endl;
    for (uint i = 0; i < input.Size(0); i++)
    {
        input.Print(i);
    }
    std::cout << "\nscale" << std::endl;
    for (uint i = 0; i < scale.Size(0); i++)
    {
        scale.Print(i);
    }
    std::cout << "\ninput_low" << std::endl;
    for (uint i = 0; i < input_low.Size(0); i++)
    {
        input_low.Print(i);
    }
    std::cout << "\ninput_range" << std::endl;
    for (uint i = 0; i < input_range.Size(0); i++)
    {
        input_range.Print(i);
    }
#endif
    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
    m_in_defs.NodeParams = &param;

    if (NameofKernel == GAUDI_KERNEL_QUANTIZE_FWD_F32)
    {
        // execute reference implementation of the kernel.
        m_in_defs.inputTensorNr = 3;
#if LOAD_FROM_JSON
        output_ref.loadData(test_output_ref);
#else
        quantize_f32_reference_implementation(input, input_low, input_range, param.levels, output_ref, NameofKernel);
#endif
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input_low);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), input_range);
    }
    else
    {
        // execute reference implementation of the kernel.
        m_in_defs.inputTensorNr = 2;
        quantize_f32_reference_implementation(input, input_low, input_range, param.levels, output_ref, NameofKernel);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input);
    }

    m_in_defs.outputTensorNr = 1;
    LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

    kernelCount = 0;
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
    kernelNames = new char *[kernelCount];
    for (unsigned i = 0; i < kernelCount; i++)
    {
        kernelNames[i] = new char[gcapi::MAX_NODE_NAME];
    }
    result = GetKernelNames(kernelNames, &kernelCount, gcapi::DEVICE_ID_GAUDI);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Can't get kernel name!! " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }

    strcpy(m_in_defs.nodeName, kernelNames[NameofKernel]);
    result = HabanaKernel(&m_in_defs, &m_out_defs);
    if (result != gcapi::GLUE_SUCCESS)
    {
        std::cout << "Glue test failed, can't load kernel " << result << std::endl;
        ReleaseKernelNames(kernelNames, kernelCount);
        return -1;
    }
    // generate and load tensor descriptors
    std::vector<TensorDescriptor> vec;
    if (NameofKernel == GAUDI_KERNEL_QUANTIZE_BWD_F32 || NameofKernel == GAUDI_KERNEL_QUANTIZE_BWD_F32)
        vec.push_back(input.GetTensorDescriptor());

    vec.push_back(input.GetTensorDescriptor());
    vec.push_back(input_low.GetTensorDescriptor());
    vec.push_back(input_range.GetTensorDescriptor());

    vec.push_back(output.GetTensorDescriptor());
    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);

#if DEBUG_PRINTF
    std::cout << "output" << std::endl;
    for (uint i = 0; i < output_ref.Size(0); i++)
    {
        output.Print(i);
    }
    std::cout << "output_ref" << std::endl;
    for (uint i = 0; i < output_ref.Size(0); i++)
    {
        output_ref.Print(i);
    }
#else
    output.Print();
    output_ref.Print();
#endif

    for (int element = 0; element < output_ref.ElementCount(); element++)
    {
        if (abs(output.Data()[element] - output_ref.Data()[element]) > 1e-6)
        {
            if (NameofKernel == GAUDI_KERNEL_QUANTIZE_FWD_F32)
                std::cout << "Quantize FWD F32 test failed!!" << std::endl;
            else
                std::cout << "Quantize BWD F32 test failed!!" << std::endl;
            return -1;
        }
    }
    if (NameofKernel == GAUDI_KERNEL_QUANTIZE_FWD_F32)
        std::cout << "Quantize FWD F32 test pass!!" << std::endl;
    else
        std::cout << "Quantize BWD F32 test pass!!" << std::endl;

    return 0;
}
