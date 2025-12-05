#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <wgpu/wgpu.h>

// ------------------------ Callbacks ------------------------
static void handle_device(WGPURequestDeviceStatus, WGPUDevice device, const char*, void* userdata) {
    *(WGPUDevice*)userdata = device;
}
static void handle_map(WGPUBufferMapAsyncStatus, void*) {}

// ------------------------ Buffer wrapper ------------------------
struct Buffer {
    WGPUBuffer buffer{};
    uint32_t size;
    Buffer(WGPUDevice device, uint32_t byte_size, WGPUBufferUsageFlags usage) : size(byte_size) {
        WGPUBufferDescriptor desc{};
        desc.size = byte_size;
        desc.usage = usage;
        desc.mappedAtCreation = false;
        buffer = wgpuDeviceCreateBuffer(device, &desc);
        assert(buffer);
    }
    ~Buffer() { wgpuBufferRelease(buffer); }
};

// ------------------------ Shader module wrapper ------------------------
struct Shader {
    WGPUShaderModule module;
    Shader(WGPUDevice device, const char* code) {
        WGPUShaderModuleWGSLDescriptor wgsl{};
        wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
        wgsl.code = code;

        WGPUShaderModuleDescriptor desc{};
        desc.nextInChain = (const WGPUChainedStruct*)&wgsl;
        module = wgpuDeviceCreateShaderModule(device, &desc);
        assert(module);
    }
    ~Shader() { wgpuShaderModuleRelease(module); }
};

// ------------------------ Compute pipeline wrapper ------------------------
struct ComputePipeline {
    WGPUComputePipeline pipeline{};
    WGPUBindGroup bind_group{};
    ComputePipeline(WGPUDevice device, Shader& shader, const std::vector<WGPUBindGroupEntry>& entries) {
        WGPUComputePipelineDescriptor desc{};
        desc.compute.module = shader.module;
        desc.compute.entryPoint = "main";
        pipeline = wgpuDeviceCreateComputePipeline(device, &desc);
        assert(pipeline);

        WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
        WGPUBindGroupDescriptor bg_desc{};
        bg_desc.layout = layout;
        bg_desc.entryCount = entries.size();
        bg_desc.entries = entries.data();
        bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);
        assert(bind_group);
    }
    ~ComputePipeline() {
        wgpuBindGroupRelease(bind_group);
        wgpuComputePipelineRelease(pipeline);
    }
};

// ------------------------ WebGPU instance ------------------------
struct WebGpu {
    WGPUInstance instance{};
    WGPUAdapter adapter{};
    WGPUDevice device{};
    WGPUQueue queue{};

    std::vector<Buffer> buffers;
    std::vector<ComputePipeline> pipelines;

    WebGpu() {
        instance = wgpuCreateInstance(nullptr);
        assert(instance);

        WGPUAdapter adapters[4];
        wgpuInstanceEnumerateAdapters(instance, nullptr, adapters);
        adapter = adapters[0];

        WGPUAdapterFeatures features = wgpuAdapterGetFeatures(adapter);
        if (features & WGPUAdapterFeature_TextureCompressionBC) {
            std::cout << "Adapter supports BC texture compression\n";
        }

        wgpuAdapterRequestDevice(adapter, nullptr, handle_device, &device);
        assert(device);

        queue = wgpuDeviceGetQueue(device);
        assert(queue);
    }

    template<typename T>
    void addBuffer(uint32_t count, WGPUBufferUsageFlags usage) {
        buffers.emplace_back(device, count * sizeof(T), usage);
    }

    void addPipeline(const char* code, const std::vector<WGPUBindGroupEntry>& entries) {
        Shader shader(device, code);
        pipelines.emplace_back(device, shader, entries);
    }

    void dispatch(uint32_t index, uint32_t x, uint32_t y = 1, uint32_t z = 1) {
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);

        wgpuComputePassEncoderSetPipeline(pass, pipelines[index].pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, pipelines[index].bind_group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, z);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
    }

    void copyBuffer(Buffer& src, Buffer& dst, uint64_t size) {
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, src.buffer, 0, dst.buffer, 0, size);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, nullptr);
        wgpuCommandEncoderRelease(encoder);
        wgpuQueueSubmit(queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
    }

    void mapRead(Buffer& buf, void* out) {
        wgpuBufferMapAsync(buf.buffer, WGPUMapMode_Read, 0, buf.size, handle_map, nullptr);
        wgpuDevicePoll(device, true, nullptr);
        memcpy(out, wgpuBufferGetMappedRange(buf.buffer, 0, buf.size), buf.size);
        wgpuBufferUnmap(buf.buffer);
    }

    ~WebGpu() {
        wgpuQueueRelease(queue);
        wgpuDeviceRelease(device);
        wgpuAdapterRelease(adapter);
        wgpuInstanceRelease(instance);
    }
};

// ------------------------ WGSL Shaders ------------------------
const char* init_shader = R"(
struct Vertices { pos: array<vec4<f32>>; };
@group(0) @binding(0) var<storage, read_write> vertices: Vertices;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    vertices.pos[i] = vec4<f32>(f32(i%32), f32(i%16), f32(i%8), 1.0);
}
)";

const char* compute_shader = R"(
struct Vertices { pos: array<vec4<f32>>; };
struct Results { data: array<f32>; };
@group(0) @binding(0) var<storage, read> vertices: Vertices;
@group(0) @binding(1) var<storage, read_write> results: Results;
@group(0) @binding(2) var<uniform> settings: vec4<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= u32(settings.w)) { return; }
    let dx = vertices.pos[i].x - settings.x;
    let dy = vertices.pos[i].y - settings.y;
    let dz = vertices.pos[i].z - settings.z;
    results.data[i] = 1.0 / sqrt(dx*dx + dy*dy + dz*dz + 0.001);
}
)";

// ------------------------ Main ------------------------
int main() {
    constexpr uint32_t N = 1024;
    WebGpu gpu;

    gpu.addBuffer<float>(N * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc); // vertices
    gpu.addBuffer<float>(N, WGPUBufferUsage_Storage | WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst);     // results
    gpu.addBuffer<float>(4, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);                              // settings

    std::vector<WGPUBindGroupEntry> init_entries = { {0, gpu.buffers[0].buffer, 0, N * sizeof(float) * 4} };
    gpu.addPipeline(init_shader, init_entries);

    std::vector<WGPUBindGroupEntry> compute_entries = {
        {0, gpu.buffers[0].buffer, 0, N * sizeof(float) * 4},
        {1, gpu.buffers[1].buffer, 0, N * sizeof(float)},
        {2, gpu.buffers[2].buffer, 0, 4 * sizeof(float)}
    };
    gpu.addPipeline(compute_shader, compute_entries);

    gpu.dispatch(0, (N+63)/64);

    float settings[4] = {10.0f, 10.0f, 10.0f, float(N)};
    wgpuQueueWriteBuffer(gpu.queue, gpu.buffers[2].buffer, 0, settings, sizeof(settings));

    gpu.dispatch(1, (N+63)/64);

    std::vector<float> results(N);
    gpu.mapRead(gpu.buffers[1], results.data());

    double total = 0;
    for(auto v : results) total += v;
    std::cout << "Total potential: " << total << "\n";
}
