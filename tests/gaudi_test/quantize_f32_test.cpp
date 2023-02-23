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
#include <torch/torch.h>
#include <regex>

#define DEBUG_PRINTF 0
#define ROOT_DIR "../test_binaries/"
#define DEFAULT_THRESHOLD 1e-06

std::vector<char> get_the_bytes(std::string filename)
{
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

std::vector<size_t> getMemDims(const int nDims, c10::ArrayRef<long int> dims)
{
    std::vector<size_t> mDims(5, 1);
    for (int i = 0; i < nDims; i++)
    {
        mDims[i] = dims[nDims - 1 - i];
    }
    return mDims;
}

int QuantizeF32Test::runTest(Gaudi_Kernel_Name_e NameofKernel)
{
    std::string fn;
    std::string config;
// Config
#if CONFIG_NUM == 1
    config = "qio_rn18-4bit-wq-sym-aq-asym-per-channel";
#elif CONFIG_NUM == 2
    config = "qio_rn18-8bit-wq-asym-aq-asym-per-tensor";
#elif CONFIG_NUM == 3
    config = "qio_rn18-8bit-wq-asym-aq-sym-per-channel";
#elif CONFIG_NUM == 4
    config = "qio_rn18-8bit-wq-sym-aq-asym-per-channel";
#elif CONFIG_NUM == 5
    config = "qio_rn18-8bit-wq-sym-aq-sym-per-tensor";
#endif
    fn.append(config + "/");
// Weight binaries
#if IS_WEIGHT
    fn.append("wt/");
#if LAYER_NUM == 1
    fn.append("ResNet/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 2
    fn.append("ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 3
    fn.append("ResNet/Sequential[layer1]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 4
    fn.append("ResNet/Sequential[layer1]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 5
    fn.append("ResNet/Sequential[layer1]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 6
    fn.append("ResNet/Sequential[layer2]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 7
    fn.append("ResNet/Sequential[layer2]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 8
    fn.append("ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 9
    fn.append("ResNet/Sequential[layer2]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 10
    fn.append("ResNet/Sequential[layer2]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 11
    fn.append("ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 12
    fn.append("ResNet/Sequential[layer3]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 13
    fn.append("ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 14
    fn.append("ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 15
    fn.append("ResNet/Sequential[layer3]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 16
    fn.append("ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 17
    fn.append("ResNet/Sequential[layer4]/BasicBlock[0]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 18
    fn.append("ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/NNCFConv2d[0]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 19
    fn.append("ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv1]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 20
    fn.append("ResNet/Sequential[layer4]/BasicBlock[1]/NNCFConv2d[conv2]/conv2d_0|WEIGHT/");
#elif LAYER_NUM == 21
    fn.append("ResNet/NNCFLinear[fc]/linear_0|WEIGHT/");
#endif
// Activation binaries
#else
    fn.append("act/");
#if LAYER_NUM == 1
    fn.append("ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 2
    fn.append("ResNet/Sequential[layer1]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 3
    fn.append("ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 4
    fn.append("ResNet/Sequential[layer1]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 5
    fn.append("ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 6
    fn.append("ResNet/Sequential[layer2]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 7
    fn.append("ResNet/Sequential[layer2]/BasicBlock[0]/Sequential[downsample]/NNCFBatchNorm2d[1]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 8
    fn.append("ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 9
    fn.append("ResNet/Sequential[layer2]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 10
    fn.append("ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 11
    fn.append("ResNet/Sequential[layer3]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 12
    fn.append("ResNet/Sequential[layer3]/BasicBlock[0]/Sequential[downsample]/NNCFBatchNorm2d[1]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 13
    fn.append("ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 14
    fn.append("ResNet/Sequential[layer3]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 15
    fn.append("ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 16
    fn.append("ResNet/Sequential[layer4]/BasicBlock[0]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 17
    fn.append("ResNet/Sequential[layer4]/BasicBlock[0]/Sequential[downsample]/NNCFBatchNorm2d[1]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 18
    fn.append("ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 19
    fn.append("ResNet/Sequential[layer4]/BasicBlock[1]/NNCFBatchNorm2d[bn2]/batch_norm_0|OUTPUT/");
#elif LAYER_NUM == 20
    fn.append("ResNet/Sequential[layer4]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT/");
#elif LAYER_NUM == 21
    fn.append("ResNet/AdaptiveAvgPool2d[avgpool]/adaptive_avg_pool2d_0|OUTPUT/");
#elif LAYER_NUM == 22
    fn.append("ResNet/ReLU[relu]/relu__0|OUTPUT/");
#elif LAYER_NUM == 23
    fn.append("ResNet/Sequential[layer1]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT/");
#elif LAYER_NUM == 24
    fn.append("ResNet/Sequential[layer1]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT/");
#elif LAYER_NUM == 25
    fn.append("ResNet/Sequential[layer2]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT/");
#elif LAYER_NUM == 26
    fn.append("ResNet/Sequential[layer2]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT/");
#elif LAYER_NUM == 27
    fn.append("ResNet/Sequential[layer3]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT/");
#elif LAYER_NUM == 28
    fn.append("ResNet/Sequential[layer3]/BasicBlock[1]/ReLU[relu]/relu__1|OUTPUT/");
#elif LAYER_NUM == 29
    fn.append("ResNet/Sequential[layer4]/BasicBlock[0]/ReLU[relu]/relu__1|OUTPUT/");
#endif
#endif
    std::cout << "[config]/[wt|act]/[layer] :\n" << fn << std::endl;

    fn.insert(0, ROOT_DIR);
    if (NameofKernel == GAUDI_KERNEL_QUANTIZE_FWD_F32)
    {
        fn.append("Forward/");
    }
    else
    {
        fn.append("Backward/");
    }

    unsigned kernelCount;
    gcapi::GlueCodeReturn_t result;
    char **kernelNames = nullptr;

    QuantizeF32::QuantizeParam param;

    auto input_pt = torch::pickle_load(get_the_bytes(fn + "/i/input_.pt")).toTensor();
    auto input_low_pt = torch::pickle_load(get_the_bytes(fn + "/i/input_low.pt")).toTensor();
    auto input_range_pt = torch::pickle_load(get_the_bytes(fn + "/i/input_range.pt")).toTensor();
    auto levels_pt = torch::pickle_load(get_the_bytes(fn + "/i/levels.pt")).toInt();

    const auto inputNumDims = input_pt.sizes().size();
    const auto inputMemDims = getMemDims(inputNumDims, input_pt.sizes());
    const auto scaleNumDims = input_range_pt.sizes().size();
    const auto scaleMemDims = getMemDims(scaleNumDims, input_range_pt.sizes());

    unsigned int fmInitializer[] = {(uint)inputMemDims[0], (uint)inputMemDims[1], (uint)inputMemDims[2], (uint)inputMemDims[3], (uint)inputMemDims[4]};
    unsigned int scaleinitializer[] = {(uint)scaleMemDims[0], (uint)scaleMemDims[1], (uint)scaleMemDims[2], (uint)scaleMemDims[3], (uint)scaleMemDims[4]};

    float_5DTensor input(fmInitializer);
    float_5DTensor input_low(scaleinitializer);
    float_5DTensor input_range(scaleinitializer);
    float_5DTensor scale(scaleinitializer);

    // For forward only
    float_5DTensor output(fmInitializer);
    float_5DTensor output_ref(fmInitializer);

    // For backward only
    float_5DTensor grad_output(fmInitializer);

    float_5DTensor grad_input(fmInitializer);
    float_5DTensor grad_input_ref(fmInitializer);

    float_5DTensor grad_input_low(scaleinitializer);
    float_5DTensor grad_input_low_ref(scaleinitializer);

    float_5DTensor grad_input_range(scaleinitializer);
    float_5DTensor grad_input_range_ref(scaleinitializer);

    input.loadData(input_pt);
    input_low.loadData(input_low_pt);
    input_range.loadData(input_range_pt);
    param.levels = levels_pt;

    bool is_sym = false;

    // generate and load tensor descriptors
    std::vector<TensorDesc> vec;

    std::cout << "input (" << input.Size(0) << "," << input.Size(1) << "," << input.Size(2) << "," << input.Size(3) << ")" << std::endl;
    std::cout << "input_low (" << input_low.Size(0) << "," << input_low.Size(1) << "," << input_low.Size(2) << "," << input_low.Size(3) << ")" << std::endl;
    std::cout << "input_range (" << input_range.Size(0) << "," << input_range.Size(1) << "," << input_range.Size(2) << "," << input_range.Size(3) << ")" << std::endl;

    if (NameofKernel == GAUDI_KERNEL_QUANTIZE_FWD_F32)
    {
        auto output_pt = torch::pickle_load(get_the_bytes(fn + "/o/output.pt")).toTensor();
        output_ref.loadData(output_pt);

        m_in_defs.inputTensorNr = 3;
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), input);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input_low);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), input_range);

        m_in_defs.outputTensorNr = 1;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), output);

        vec.push_back(input.GetTensorDescriptor());
        vec.push_back(input_low.GetTensorDescriptor());
        vec.push_back(input_range.GetTensorDescriptor());
        vec.push_back(output.GetTensorDescriptor());
    }
    else
    {
        auto grad_output_pt = torch::pickle_load(get_the_bytes(fn + "/i/grad_output.pt")).toTensor();
        auto level_high_pt = torch::pickle_load(get_the_bytes(fn + "/i/level_high.pt")).toInt();
        auto level_low_pt = torch::pickle_load(get_the_bytes(fn + "/i/level_low.pt")).toInt();

        is_sym = (fn.substr(fn.find((IS_WEIGHT) ? "wq" : "aq"), 4)[3] == 's') ? true : false;

        auto grad_input_pt = torch::pickle_load(get_the_bytes(fn + "/o/grad_input.pt")).toTensor();

        grad_output.loadData(grad_output_pt);
        param.level_low = level_low_pt;
        param.level_high = level_high_pt;

        grad_input_ref.loadData(grad_input_pt);

        std::cout << "grad_output (" << grad_output.Size(0) << "," << grad_output.Size(1) << "," << grad_output.Size(2) << "," << grad_output.Size(3) << ")" << std::endl;
        std::cout << "is_sym :\t" << is_sym << std::endl
                  << std::endl;
        m_in_defs.inputTensorNr = 4;
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[0]), grad_output);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[1]), input);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[2]), input_low);
        LoadTensorToGcDescriptor(&(m_in_defs.inputTensors[3]), input_range);

        if (is_sym)
        {
            auto grad_input_range_pt = torch::pickle_load(get_the_bytes(fn + "/o/grad_scale.pt")).toTensor();
            grad_input_range_ref.loadData(grad_input_range_pt);
        }
        else
        {
            auto grad_input_low_pt = torch::pickle_load(get_the_bytes(fn + "/o/grad_input_low.pt")).toTensor();
            grad_input_low_ref.loadData(grad_input_low_pt);

            auto grad_input_range_pt = torch::pickle_load(get_the_bytes(fn + "/o/grad_input_range.pt")).toTensor();
            grad_input_range_ref.loadData(grad_input_range_pt);
        }
        m_in_defs.outputTensorNr = 3;
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[0]), grad_input);
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[1]), grad_input_low);
        LoadTensorToGcDescriptor(&(m_in_defs.outputTensors[2]), grad_input_range);

        vec.push_back(grad_output.GetTensorDescriptor());
        vec.push_back(input.GetTensorDescriptor());
        vec.push_back(input_low.GetTensorDescriptor());
        vec.push_back(input_range.GetTensorDescriptor());
        vec.push_back(grad_input.GetTensorDescriptor());
        vec.push_back(grad_input_low.GetTensorDescriptor());
        vec.push_back(grad_input_range.GetTensorDescriptor());
    }

    // generate input for query call
    m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;
    m_in_defs.NodeParams = &param;

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

    // execute a simulation of the kernel using TPC simulator,
    TestBase::RunSimulation(vec, m_in_defs, m_out_defs);
    ReleaseKernelNames(kernelNames, kernelCount);

#if DEBUG_PRINTF
    std::cout << "output" << std::endl;
    for (uint i = 0; i < output.Size(0); i++)
    {
        output.Print(i);
    }
    std::cout << "output_ref" << std::endl;
    for (uint i = 0; i < output_ref.Size(0); i++)
    {
        output_ref.Print();
    }
#endif
    int mismatched = 0;
    float threshold = DEFAULT_THRESHOLD;
    float output_min = 0.0f, output_max = 0.0f, ref_min = 0.0f, ref_max = 0.0f, max_abs = 0.0f, max_rel = 0.0f;
    int max_abs_elem = 0;
    bool test_failed = false;

    if (NameofKernel == GAUDI_KERNEL_QUANTIZE_FWD_F32)
    {
        for (int element = 0; element < output_ref.ElementCount(); element++)
        {
            output_min = std::min(output_min, output.Data()[element]);
            output_max = std::max(output_max, output.Data()[element]);
            ref_min = std::min(ref_min, output_ref.Data()[element]);
            ref_max = std::max(ref_max, output_ref.Data()[element]);
            if (abs(output.Data()[element] - output_ref.Data()[element]) > max_abs)
                max_abs_elem = element;
            max_abs = std::max(max_abs, abs(output.Data()[element] - output_ref.Data()[element]));
            max_rel = std::max(max_rel, (output.Data()[element] - output_ref.Data()[element]) / output_ref.Data()[element]);
            if (abs(output.Data()[element] - output_ref.Data()[element]) > threshold || !(std::isnan(output.Data()[element]) == std::isnan(output_ref.Data()[element])))
            {
                mismatched++;
            }
        }
        std::cout << "Threshold :" << threshold << "\tMismatched found : " << mismatched << std::endl;
        std::cout << "Min Value:" << std::endl;
        std::cout << "\toutput\t:" << output_min << std::endl;
        std::cout << "\tref\t:" << ref_min << std::endl;
        std::cout << "Max Value:" << std::endl;
        std::cout << "\toutput\t:" << output_max << std::endl;
        std::cout << "\tref\t:" << ref_max << std::endl;
        std::cout << "Element (0-index)\t:" << max_abs_elem << std::endl;
        std::cout << "Max Abs\t:" << max_abs << std::endl;
        std::cout << "Max Rel\t:" << max_rel << std::endl;
        test_failed |= mismatched;
    }
    else
    {
        std::cout << "===========" << std::endl;
        std::cout << "grad_input" << std::endl;
        std::cout << "===========" << std::endl;
        for (int element = 0; element < grad_input_ref.ElementCount(); element++)
        {
            output_min = std::min(output_min, grad_input.Data()[element]);
            output_max = std::max(output_max, grad_input.Data()[element]);
            ref_min = std::min(ref_min, grad_input_ref.Data()[element]);
            ref_max = std::max(ref_max, grad_input_ref.Data()[element]);
            if (abs(grad_input.Data()[element] - grad_input_ref.Data()[element]) > max_abs)
                max_abs_elem = element;
            max_abs = std::max(max_abs, abs(grad_input.Data()[element] - grad_input_ref.Data()[element]));
            max_rel = std::max(max_rel, (grad_input.Data()[element] - grad_input_ref.Data()[element]) / grad_input_ref.Data()[element]);
            if (abs(grad_input.Data()[element] - grad_input_ref.Data()[element]) > threshold)
            {
                mismatched++;
            }
        }
        std::cout << "Threshold :" << threshold << "\tMismatched found : " << mismatched << std::endl;
        std::cout << "Min Value:" << std::endl;
        std::cout << "\toutput\t:" << output_min << std::endl;
        std::cout << "\tref\t:" << ref_min << std::endl;
        std::cout << "Max Value:" << std::endl;
        std::cout << "\toutput\t:" << output_max << std::endl;
        std::cout << "\tref\t:" << ref_max << std::endl;
        std::cout << "Element (0-index)\t:" << max_abs_elem << std::endl;

        std::cout << "Max Abs\t:" << max_abs << std::endl;
        std::cout << "Max Rel\t:" << max_rel << std::endl;
        test_failed |= mismatched;

        if (!is_sym)
        {
            output_min = output_max = ref_min = ref_max = max_abs, max_rel = 0.0f;
            mismatched = 0;
            max_abs_elem = 0;
            std::cout << "==============" << std::endl;
            std::cout << "grad_input_low" << std::endl;
            std::cout << "==============" << std::endl;
            for (int element = 0; element < grad_input_low_ref.ElementCount(); element++)
            {
                output_min = std::min(output_min, grad_input_low.Data()[element]);
                output_max = std::max(output_max, grad_input_low.Data()[element]);
                ref_min = std::min(ref_min, grad_input_low_ref.Data()[element]);
                ref_max = std::max(ref_max, grad_input_low_ref.Data()[element]);
                if (abs(grad_input_low.Data()[element] - grad_input_low_ref.Data()[element]) > max_abs)
                    max_abs_elem = element;
                max_abs = std::max(max_abs, abs(grad_input_low.Data()[element] - grad_input_low_ref.Data()[element]));
                max_rel = std::max(max_rel, (grad_input_low.Data()[element] - grad_input_low_ref.Data()[element]) / grad_input_low_ref.Data()[element]);
                if (abs(grad_input_low.Data()[element] - grad_input_low_ref.Data()[element]) > threshold)
                {
                    mismatched++;
                }
            }
            std::cout << "Threshold :" << threshold << "\tMismatched found : " << mismatched << std::endl;
            std::cout << "Min Value:" << std::endl;
            std::cout << "\toutput\t:" << output_min << std::endl;
            std::cout << "\tref\t:" << ref_min << std::endl;
            std::cout << "Max Value:" << std::endl;
            std::cout << "\toutput\t:" << output_max << std::endl;
            std::cout << "\tref\t:" << ref_max << std::endl;
            std::cout << "Element (0-index)\t:" << max_abs_elem << std::endl;

            std::cout << "Max Abs\t:" << max_abs << std::endl;
            std::cout << "Max Rel\t:" << max_rel << std::endl;
            test_failed |= mismatched;
        }

        output_min = output_max = ref_min = ref_max = max_abs, max_rel = 0.0f;
        mismatched = 0;
        max_abs_elem = 0;
        std::cout << "==============" << std::endl;
        if (is_sym)
        {
            std::cout << "grad_scale" << std::endl;
        }
        else
        {
            std::cout << "grad_input_range" << std::endl;
        }
        std::cout << "==============" << std::endl;
        for (int element = 0; element < grad_input_range_ref.ElementCount(); element++)
        {
            output_min = std::min(output_min, grad_input_range.Data()[element]);
            output_max = std::max(output_max, grad_input_range.Data()[element]);
            ref_min = std::min(ref_min, grad_input_range_ref.Data()[element]);
            ref_max = std::max(ref_max, grad_input_range_ref.Data()[element]);
            if (abs(grad_input_range.Data()[element] - grad_input_range_ref.Data()[element]) > max_abs)
                max_abs_elem = element;
            max_abs = std::max(max_abs, abs(grad_input_range.Data()[element] - grad_input_range_ref.Data()[element]));
            max_rel = std::max(max_rel, (grad_input_range.Data()[element] - grad_input_range_ref.Data()[element]) / grad_input_range_ref.Data()[element]);
            if (abs(grad_input_range.Data()[element] - grad_input_range_ref.Data()[element]) > threshold)
            {
                mismatched++;
            }
        }
        std::cout << "Threshold :" << threshold << "\tMismatched found : " << mismatched << std::endl;
        std::cout << "Min Value:" << std::endl;
        std::cout << "\toutput\t:" << output_min << std::endl;
        std::cout << "\tref\t:" << ref_min << std::endl;
        std::cout << "Max Value:" << std::endl;
        std::cout << "\toutput\t:" << output_max << std::endl;
        std::cout << "\tref\t:" << ref_max << std::endl;
        std::cout << "Element (0-index)\t:" << max_abs_elem << std::endl;

        std::cout << "Max Abs\t:" << max_abs << std::endl;
        std::cout << "Max Rel\t:" << max_rel << std::endl;
        test_failed |= mismatched;
    }

    if (NameofKernel == GAUDI_KERNEL_QUANTIZE_FWD_F32)
        if (test_failed)
        {
            std::cout << "Quantize FWD F32 test failed!!" << std::endl;
            return -1;
        }
        else
        {
            std::cout << "Quantize FWD F32 test pass!!" << std::endl;
            return 0;
        }
    else if (test_failed)
    {
        std::cout << "Quantize BWD F32 test failed!!" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Quantize BWD F32 test pass!!" << std::endl;
        return 0;
    }
}
