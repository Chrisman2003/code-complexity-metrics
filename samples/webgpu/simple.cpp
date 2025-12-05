#include <iostream>
#include <vector>
#include <wgpu/wgpu.h>
#include <cassert>

static void handle_device(WGPURequestDeviceStatus, WGPUDevice device, const char*, void* userdata) {
    *(WGPUDevice*)userdata = device;
}
static void handle_map(WGPUBufferMapAsyncStatus, void*) {}

int main() {
    constexpr uint32_t N = 16;

    // 1. Create instance, adapter, device, queue
    WGPUInstance instance = wgpuCreateInstance(nullptr);
    WGPUAdapter adapters[1];
    wgpuInstanceEnumerateAdapters(instance, nullptr, adapters);
    WGPUDevice device = nullptr;
    wgpuAdapterRequestDevice(adapters[0], nullptr, handle_device, &device);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // 2. Create buffers
    WGPUBuffer input = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor{
        .size = N * sizeof(float),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        .mappedAtCreation = false
    }));
    WGPUBuffer output = wgpuDeviceCreateBuffer(device, &(WGPUBufferDescriptor{
        .size = N * sizeof(float),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
        .mappedAtCreation = false
    }));

    // 3. Compute shader (WGSL)
    const char* shader_code = R"(
        struct Data { numbers: array<f32>; };
        @group(0) @binding(0) var<storage, read_write> input: Data;
        @group(0) @binding(1) var<storage, read_write> output: Data;

        @compute @workgroup_size(16)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let idx = id.x;
            output.numbers[idx] = input.numbers[idx] * input.numbers[idx];
        }
    )";

    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device, &(WGPUShaderModuleDescriptor{
        .nextInChain = (const WGPUChainedStruct*)&(WGPUShaderModuleWGSLDescriptor{
            .chain = {.sType = WGPUSType_ShaderModuleWGSLDescriptor},
            .code = shader_code
        })
    }));

    WGPUBindGroupEntry entries[2] = {
        {.binding = 0, .buffer = input, .offset = 0, .size = N * sizeof(float)},
        {.binding = 1, .buffer = output, .offset = 0, .size = N * sizeof(float)}
    };

    WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(device, &(WGPUBindGroupLayoutDescriptor{
        .entryCount = 2,
        .entries = (WGPUBindGroupLayoutEntry[]){
            {.binding=0,.visibility=WGPUShaderStage_Compute,.buffer={.type=WGPUBufferBindingType_Storage}},
            {.binding=1,.visibility=WGPUShaderStage_Compute,.buffer={.type=WGPUBufferBindingType_Storage}}
        }
    }));
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &(WGPUBindGroupDescriptor{.layout=layout,.entryCount=2,.entries=entries}));

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &(WGPUComputePipelineDescriptor{
        .compute = {.module = shader, .entryPoint = "main"}
    }));

    // 4. Initialize input
    std::vector<float> data(N);
    for (uint32_t i=0;i<N;i++) data[i] = float(i);
    wgpuQueueWriteBuffer(queue, input, 0, data.data(), N*sizeof(float));

    // 5. Encode and dispatch compute
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuCommandEncoderRelease(encoder);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);

    // 6. Read back results
    wgpuBufferMapAsync(output, WGPUMapMode_Read, 0, N*sizeof(float), handle_map, nullptr);
    wgpuDevicePoll(device, true, nullptr);
    auto* result = (float*)wgpuBufferGetMappedRange(output, 0, N*sizeof(float));

    std::cout << "Squares: ";
    for (uint32_t i=0;i<N;i++) std::cout << result[i] << " ";
    std::cout << "\n";
    wgpuBufferUnmap(output);

    return 0;
}
