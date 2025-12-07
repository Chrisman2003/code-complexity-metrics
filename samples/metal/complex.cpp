#include <Metal/Metal.hpp>
#include <iostream>
#include <vector>
#include <stdexcept>

int main() {
    using float_type = float;

    // --- 1. Initialize Metal device ---
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal not supported.\n";
        return 1;
    }

    // --- 2. Create shader source ---
    const char* shaderSource = R"(
        #include <metal_stdlib>
        using namespace metal;

        kernel void vec_add(const device float* inA [[buffer(0)]],
                            const device float* inB [[buffer(1)]],
                            device float* out  [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
            out[id] = inA[id] + inB[id];
        }
    )";

    // --- 3. Compile shader into library ---
    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(
        NS::String::string(shaderSource, NS::UTF8StringEncoding),
        nullptr,
        &error
    );

    if (!library) {
        std::cerr << "Shader compile error: "
                  << error->localizedDescription()->utf8String() << "\n";
        return 1;
    }

    // --- 4. Create compute pipeline ---
    MTL::Function* function =
        library->newFunction(NS::String::string("vec_add", NS::UTF8StringEncoding));

    MTL::ComputePipelineState* pipeline = device->newComputePipelineState(function, &error);
    if (!pipeline) {
        std::cerr << "Pipeline creation error: "
                  << error->localizedDescription()->utf8String() << "\n";
        return 1;
    }

    // --- 5. Command queue ---
    MTL::CommandQueue* queue = device->newCommandQueue();

    // --- 6. Prepare input data ---
    const size_t N = 16;
    std::vector<float_type> A(N), B(N), C(N, 0);

    for (size_t i = 0; i < N; ++i) {
        A[i] = float_type(i);
        B[i] = float_type(i * 2);
    }

    auto bufA = device->newBuffer(A.data(), N * sizeof(float_type), MTL::ResourceStorageModeShared);
    auto bufB = device->newBuffer(B.data(), N * sizeof(float_type), MTL::ResourceStorageModeShared);
    auto bufC = device->newBuffer(C.data(), N * sizeof(float_type), MTL::ResourceStorageModeShared);

    // --- 7. Encode GPU work ---
    auto cmd = queue->commandBuffer();
    auto enc = cmd->computeCommandEncoder();

    enc->setComputePipelineState(pipeline);
    enc->setBuffer(bufA, 0, 0);
    enc->setBuffer(bufB, 0, 1);
    enc->setBuffer(bufC, 0, 2);

    MTL::Size gridSize(N, 1, 1);
    MTL::Size threadgroupSize(
        std::min<size_t>(pipeline->maxTotalThreadsPerThreadgroup(), N),
        1,
        1
    );

    enc->dispatchThreads(gridSize, threadgroupSize);
    enc->endEncoding();

    cmd->commit();
    cmd->waitUntilCompleted();

    // --- 8. Copy results ---
    memcpy(C.data(), bufC->contents(), sizeof(float_type) * N);

    // --- 9. Print results ---
    std::cout << "Result of A[i] + B[i]:\n";
    for (auto x : C) std::cout << x << " ";
    std::cout << "\n";

    // --- 10. Cleanup ---
    bufA->release();
    bufB->release();
    bufC->release();
    enc->release();
    pipeline->release();
    function->release();
    library->release();
    queue->release();
    device->release();

    return 0;
}
