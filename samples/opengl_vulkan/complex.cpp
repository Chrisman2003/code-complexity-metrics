#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

static const uint32_t compute_shader[] = {
#include "shader/compute_comp.hpp"
};
const size_t compute_shader_len = sizeof(compute_shader);

constexpr uint32_t WORKGROUP_SIZE = 64;

struct PushConstants {
    glm::vec3 scale;
};

class VulkanOpenGLHybrid {
public:
    VulkanOpenGLHybrid() {
        initGLFW();
        initVulkan();
        createBuffers();
        createDescriptorSets();
        createPipeline();
        allocateCommandBuffer();
    }

    ~VulkanOpenGLHybrid() {
        if (_window) glfwDestroyWindow(_window);
        glfwTerminate();
    }

    void run() {
        // Record commands
        _cmdBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        _cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *_pipeline);
        _cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *_pipelineLayout, 0, {*_descriptorSets[0]}, {});
        PushConstants pc{ glm::vec3(1.5f, 2.0f, 0.5f) };
        _cmdBuffer.pushConstants(*_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushConstants), &pc);
        _cmdBuffer.dispatch((_data.size() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE, 1, 1);
        _cmdBuffer.end();

        _queue.submit({{nullptr, nullptr, *_cmdBuffer}}, *_fence);
        _device.waitForFences(*_fence, true, uint64_t(-1));

        copyVulkanBufferToOpenGL();

        renderLoop();
    }

private:
    void initGLFW() {
        if (!glfwInit()) throw std::runtime_error("GLFW init failed");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        _window = glfwCreateWindow(800, 600, "Vulkan + OpenGL Hybrid", nullptr, nullptr);
        if (!_window) throw std::runtime_error("GLFW window creation failed");
        glfwMakeContextCurrent(_window);
    }

    void initVulkan() {
        vk::ApplicationInfo appInfo("HybridCompute", 1, "NoEngine", 1, VK_API_VERSION_1_1);
        _instance = vk::raii::Instance(_context, { {}, &appInfo });

        auto physical = _instance.enumeratePhysicalDevices().front();
        uint32_t queueIndex = 0;
        auto qprops = physical.getQueueFamilyProperties();
        for (uint32_t i = 0; i < qprops.size(); ++i)
            if (qprops[i].queueFlags & vk::QueueFlagBits::eCompute) { queueIndex = i; break; }

        float priority = 1.0f;
        _device = physical.createDevice({ {}, 1, &vk::DeviceQueueCreateInfo({}, queueIndex, 1, &priority) });
        _queue = _device.getQueue(queueIndex, 0);
    }

    void createBuffers() {
        // Initialize data
        _data.resize(1024, glm::vec4(0.0f));
        size_t buffer_size = _data.size() * sizeof(glm::vec4);

        vk::BufferCreateInfo bufferInfo({}, buffer_size, vk::BufferUsageFlagBits::eStorageBuffer);
        _buffer = _device.createBuffer(bufferInfo);

        auto memReq = _buffer.getMemoryRequirements();
        uint32_t memTypeIndex = findMemoryType(_instance.enumeratePhysicalDevices().front(), memReq.memoryTypeBits);
        _memory = vk::raii::DeviceMemory(_device, { memReq.size, memTypeIndex });
        _buffer.bindMemory(*_memory, 0);
    }

    void createDescriptorSets() {
        vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
        _descriptorSetLayout = _device.createDescriptorSetLayout({ {}, 1, &binding });
        _pipelineLayout = _device.createPipelineLayout({ {}, 1, &_descriptorSetLayout });

        vk::DescriptorPoolSize poolSize(vk::DescriptorType::eStorageBuffer, 1);
        _descriptorPool = _device.createDescriptorPool({ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, 1, &poolSize });

        _descriptorSets = _device.allocateDescriptorSets({ *_descriptorPool, 1, &_descriptorSetLayout });
        vk::DescriptorBufferInfo bufInfo(*_buffer, 0, VK_WHOLE_SIZE);
        _device.updateDescriptorSets({ { *_descriptorSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufInfo } }, {});
    }

    void createPipeline() {
        _shaderModule = _device.createShaderModule({ {}, compute_shader_len, compute_shader });
        vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, *_shaderModule, "main");
        _pipeline = _device.createComputePipeline({}, stage, *_pipelineLayout);
    }

    void allocateCommandBuffer() {
        _cmdPool = _device.createCommandPool({ {} , 0});
        _cmdBuffer = _device.allocateCommandBuffers({ *_cmdPool, vk::CommandBufferLevel::ePrimary, 1 })[0];
        _fence = _device.createFence({});
    }

    void copyVulkanBufferToOpenGL() {
        // Placeholder for buffer copy to OpenGL texture or VBO
        // In a real application, you could use vkBindBufferMemory + GL buffer sharing extensions
        std::cout << "Buffer ready for OpenGL rendering (simulated copy)." << std::endl;
    }

    void renderLoop() {
        while (!glfwWindowShouldClose(_window)) {
            glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Draw OpenGL scene using data produced by Vulkan compute
            // (actual OpenGL rendering code omitted for brevity)

            glfwSwapBuffers(_window);
            glfwPollEvents();
        }
    }

    uint32_t findMemoryType(const vk::raii::PhysicalDevice& physical, uint32_t bits) {
        auto memProps = physical.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
            if ((bits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible))
                return i;
        return 0;
    }

private:
    GLFWwindow* _window = nullptr;

    vk::raii::Context _context;
    vk::raii::Instance _instance;
    vk::raii::Device _device;
    vk::raii::Queue _queue;

    std::vector<glm::vec4> _data;

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
    try {
        VulkanOpenGLHybrid app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
