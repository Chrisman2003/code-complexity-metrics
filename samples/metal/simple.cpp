#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <iostream>

int main() {
    // 1) Create device and queue
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* queue = device->newCommandQueue();

    // 2) Minimal shader: add two floats
    const char* shader = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void add(const device float* A [[buffer(0)]],
                    const device float* B [[buffer(1)]],
                    device float* C [[buffer(2)]],
                    uint id [[thread_position_in_grid]])
    {
        C[id] = A[id] + B[id];
    }
    )";

    NS::Error* error = nullptr;
    auto library = device->newLibrary(NS::String::string(shader, NS::UTF8StringEncoding), nullptr, &error);
    auto func = library->newFunction(NS::String::string("add", NS::UTF8StringEncoding));
    auto pipeline = device->newComputePipelineState(func, &error);

    // 3) Buffers
    float A = 2.0f, B = 3.0f, C = 0.0f;
    auto bufA = device->newBuffer(&A, sizeof(float), MTL::ResourceStorageModeShared);
    auto bufB = device->newBuffer(&B, sizeof(float), MTL::ResourceStorageModeShared);
    auto bufC = device->newBuffer(&C, sizeof(float), MTL::ResourceStorageModeShared);

    // 4) Encode command
    auto cmd = queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();
    enc->setComputePipelineState(pipeline);
    enc->setBuffer(bufA, 0, 0);
    enc->setBuffer(bufB, 0, 1);
    enc->setBuffer(bufC, 0, 2);
    enc->dispatchThreads(MTL::Size(1,1,1), MTL::Size(1,1,1));
    enc->endEncoding();
    cmd->commit();
    cmd->waitUntilCompleted();

    // 5) Print result
    std::cout << "Result: " << C << "\n"; // still 0 in CPU memory
    memcpy(&C, bufC->contents(), sizeof(float));
    std::cout << "C = A + B = " << C << "\n";

    // 6) Cleanup
    bufA->release(); bufB->release(); bufC->release();
    pipeline->release(); func->release(); library->release(); queue->release(); device->release();

    return 0;
}
