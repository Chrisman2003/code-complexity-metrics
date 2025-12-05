#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <iostream>
#include <vector>

static const uint32_t simple_compute_shader[] = {
#include "shader/simple_comp.hpp" // small SPIR-V bytecode
};
const size_t simple_compute_shader_len = sizeof(simple_compute_shader);

constexpr uint32_t WORKGROUP_SIZE = 32;

struct PushConstants {
    float value;
};

class SimpleCompute {
public:
    SimpleCompute() {
        vk::ApplicationInfo appInfo("SimpleCompute", 1, "NoEngine", 1, VK_API_VERSION_1_1);
        _instance = vk::raii::Instance(_context, vk::InstanceCreateInfo({}, &appInfo));

        auto physical = _instance.enumeratePhysicalDevices().front();
        uint32_t queueIndex = 0;
        for (auto i = 0u; i < physical.getQueueFamilyProperties().size(); ++i) {
            if (physical.getQueueFamilyProperties()[i].queueFlags & vk::QueueFlagBits::eCompute) {
                queueIndex = i;
                break;
            }
        }

        float priority = 1.0f;
        _device = physical.createDevice(
            vk::DeviceCreateInfo({}, 1, &vk::DeviceQueueCreateInfo({}, queueIndex, 1, &priority))
        );
        _queue = _device.getQueue(queueIndex, 0);

        // Create buffer
        vk::BufferCreateInfo bufferInfo({}, 1024, vk::BufferUsageFlagBits::eStorageBuffer);
        _buffer = _device.createBuffer(bufferInfo);

        auto memReq = _buffer.getMemoryRequirements();
        uint32_t memTypeIndex = findMemoryType(physical, memReq.memoryTypeBits);
        _memory = vk::raii::DeviceMemory(_device, vk::MemoryAllocateInfo(memReq.size, memTypeIndex));
        _buffer.bindMemory(*_memory, 0);

        // Descriptor set
        vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eStorageBuffer, 1,
                                               vk::ShaderStageFlagBits::eCompute);
        _descriptorSetLayout = _device.createDescriptorSetLayout({{}, 1, &binding});
        _pipelineLayout = _device.createPipelineLayout({{}, 1, &_descriptorSetLayout});
        _descriptorPool = _device.createDescriptorPool({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 1, &vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 1)});
        _descriptorSets = _device.allocateDescriptorSets({_descriptorPool, 1, &_descriptorSetLayout});

        vk::DescriptorBufferInfo bufferDesc(*_buffer, 0, VK_WHOLE_SIZE);
        _device.updateDescriptorSets({{*_descriptorSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferDesc}}, {});

        // Shader and pipeline
        _shaderModule = _device.createShaderModule({{}, simple_compute_shader_len, simple_compute_shader});
        _pipeline = _device.createComputePipeline({}, {vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute, *_shaderModule, "main"}, *_pipelineLayout);

        // Command buffer
        _cmdPool = _device.createCommandPool({{}, queueIndex});
        _cmdBuffer = _device.allocateCommandBuffers({*_cmdPool, vk::CommandBufferLevel::ePrimary, 1})[0];
        _cmdBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        _cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *_pipeline);
        _cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *_pipelineLayout, 0, {*_descriptorSets[0]}, {});
        PushConstants pc{42.0f};
        _cmdBuffer.pushConstants(*_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants), &pc);
        _cmdBuffer.dispatch(1, 1, 1);
        _cmdBuffer.end();

        _fence = _device.createFence({});
        _queue.submit({{nullptr, nullptr, *_cmdBuffer}}, *_fence);
        _device.waitForFences(*_fence, true, uint64_t(-1));
    }

private:
    uint32_t findMemoryType(const vk::raii::PhysicalDevice& physical, uint32_t bits) {
        auto memProps = physical.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((bits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible))
                return i;
        }
        return 0;
    }

    vk::raii::Context _context;
    vk::raii::Instance _instance;
    vk::raii::Device _device;
    vk::raii::Queue _queue;

    vk::raii::Buffer _buffer;
    vk::raii::DeviceMemory _memory;

    vk::raii::DescriptorSetLayout _descriptorSetLayout;
    vk::raii::PipelineLayout _pipelineLayout;
    vk::raii::DescriptorPool _descriptorPool;
    std::vector<vk::raii::DescriptorSet> _descriptorSets;

    vk::raii::ShaderModule _shaderModule;
    vk::raii::Pipeline _pipeline;

    vk::raii::CommandPool _cmdPool;
    vk::raii::CommandBuffer _cmdBuffer;
    vk::raii::Fence _fence;
};

int main() {
    SimpleCompute compute;
    std::cout << "Simple Vulkan compute done!" << std::endl;
    return 0;
}
