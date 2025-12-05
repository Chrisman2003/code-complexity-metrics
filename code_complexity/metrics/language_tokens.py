# ------------------------------
# 0) Automated Suffixation completion
# ------------------------------
# Substring Suffix Extension Patterns with Respect to the Kleene Operator (*)
# Trailing Commas for Maintainability and minimizing potential errors
pattern_rules = {
    # All constructs with the prefixes below are matched. This includes enum error types like "cudaSuccess".
    # The idea behind this is to represent a given parallel framework with an operator-oriented lens.
    'cuda': [
        r'\b__\w+__\b',                  # CUDA qualifiers like __global__, __device__
        r'\bcuda[A-Z]\w*\b',             # CUDA runtime APIs like cudaMalloc, cudaMemcpy
        r'\batomic[A-Z]\w*\b',           # CUDA atomic intrinsics
    ],
    'opencl': [
        r'\bcl[A-Z]\w*\b',               # clCreateBuffer, clEnqueueNDRangeKernel
        r'\bget_(?:global|local|group)_id\b',  # non-capturing group
        r'\bcl(?:::\w+)+\b',
    ],
    'kokkos': [ 
        r'\bKokkos::\w+\b',              # All Kokkos namespace calls
    ],
    'openmp': [
        r'\bomp_[a-zA-Z_]+\b',           # matches OpenMP functions like omp_get_num_threads
        r'#pragma\s+omp\s+[a-zA-Z_\s]+', # matches pragmas like #pragma omp parallel for
    ],
    'adaptivecpp': [
        r'\bsycl::\w+\b',                # all SYCL class/function names
        r'\bqueue\b',                    # short forms when using namespace sycl
        r'\bparallel_for\b',
        r'\bsingle_task\b',
        r'\bnd_range\b',
        r'\bnd_item\b',
    ],
    'openacc': [
        r'\bacc_\w+\b',                  # OpenACC runtime functions like acc_malloc
        r'#pragma\s+acc\s+[a-zA-Z_\s]+', # OpenACC pragma lines
    ],
    'opengl_vulkan': [
        r'\bvk\w+\b',                    # Vulkan functions like vkCreateInstance
        r'\bgl\w+\b',                    # OpenGL functions like glBindBuffer
        r'\bVK_[A-Z0-9_]+\b',            # Vulkan constants like VK_SUCCESS
        r'\bGL_[A-Z0-9_]+\b',            # OpenGL constants like GL_COMPUTE_SHADER
        r'\bvk::\w+\b',                  # Vulkan C++ API like vk::Instance
        r'\bvk(?:::\w+)+\b',             # Vulkan C++ API like vk::Instance::Instance
        # TODO: Generalized matching for _vk_context ?
    ],
    'webgpu': [
        r'\bwgpu::\w+\b',                # all WebGPU C++ API classes
        r'\bwgpu[A-Z]\w*\b',             # WebGPU runtime functions like wgpuCreateInstance
        r'\bWGPU_[A-Z0-9_]+\b',          # constants
    ],
    'boost': [
        r'\bboost::compute::\w+\b',      # all Boost.Compute classes & functions
        r'\bBOOST_COMPUTE_FUNCTION\b',   # macro
    ],
    'metal': [
        r'\bMTL\w+\b',                   # Metal classes / types
        r'\bMTL::\w+\b',                 # Metal classes / types
        r'\b(device|thread|threadgroup|constant|kernel|sampler|texture)\b',
        r'\bdispatchThreads\b|\bdispatchThreadgroups\b|\bcommit\b|\benqueue\b',
        r'\bnew\w+With\w*:\b',           # Objective-C style method calls
        r'\bMTL_[A-Z0-9_]+\b'            # Metal constants / enums
    ],
    'thrust': [
        r'\bthrust::\w+\b',              # all Thrust API calls and classes
        r'\bTHRUST_[A-Z0-9_]+\b',        # macros
        r'\bthrust(?:::\w+)+\b',         # nested namespaces like thrust::system::cuda
    ], 
}

# ------------------------------
# 1 Standard C++ keywords
# ------------------------------
cpp_control = {
    'if', 'else', 'switch', 'case', 'for', 'while', 'do', 'break', 'continue', 'return', 'goto',
    'try', 'catch', 'throw',
    'default', 'if constexpr', 'co_return', 'requires_clause', 'requires_expression', 'synchronized',
    'namespace', 'new', 'delete', 'not', 'not_eq', 'or', 'or_eq', 'private', 'protected',
    'public', 'friend', 'virtual', 'explicit', 'inline', 'mutable', 'this', 'alignas', 'alignof', 'decltype',
    'constexpr', 'noexcept', 'co_await', 'co_yield', 'export', 'import', 'module', 'requires',
    'concept', 'asm', 'override', 'final', 'typeid', 'dynamic_cast', 'reinterpret_cast',
    'static_assert', 'thread_local', 'typename', 'sizeof', 'decltype(auto)',
    'char8_t', 'char16_t', 'char32_t', 'wchar_t', 'bool', 'signed', 'unsigned',
    'short', 'long', 'int', 'float', 'double', 'void', 'auto', 'nullptr_t', 'ptrdiff_t', 'size_t',
    'module_partition', 'concept_map', 'transaction_safe'
} # Pointer Dereferencing ?

cpp_types = {
    'char',
    'class', 'struct', 'union', 'enum', 'typedef', 'using',
    'std::string', 'std::wstring', 'std::u16string', 'std::u32string', 'std::vector', 'std::list', 'std::map',
    'std::set', 'std::unordered_map', 'std::unordered_set', 'std::deque', 'std::array', 'std::tuple', 'std::pair',
    'std::function', 'std::shared_ptr', 'std::unique_ptr', 'std::weak_ptr', 'std::optional', 'std::variant', 'std::any',
    'std::span', 'std::byte', 'std::chrono::duration', 'std::chrono::time_point', 'std::chrono::system_clock',
    'std::chrono::steady_clock', 'std::chrono::high_resolution_clock', 'std::filesystem::path',
    'std::initializer_list', 'std::atomic'
}

cpp_modifiers = {
    'template', 'static', 'const',
    'volatile', 'operator', 'extern',
    'register', 'type',
    'consteval', 'constinit', 'static_cast', 'const_cast',
    'false', 'true', 'nullptr', 'likely', 'unlikely', 'nodiscard', 'maybe_unused'
}

cpp_operators = {
    '-', '+', '*', '/', '%', '=', '==', '!=', '<=', '>=', '<', '>',
    '&', '|', '^', '~', '!', '+=', '-=', '*=', '/=', '%=', '<<=',
    '>>=', '&=', '|=', '^=', '>>', '<<', '&&', '||', '++', '--',
    '->', '->*', '.', '::', '(', ')', '{', '}',
    '[', ']', ',', ':', ';', '#', '@', '...', '?',
    'and', 'bitand', 'bitor', 'xor', 'compl', 'and_eq', 'xor_eq',
    '<=>', 'co_await_operator', 'operator<=>'
}

cpp_side_effect_functions = {
    'printf', 'fprintf', 'sprintf', 'snprintf', 'puts', 'putchar', 'scanf', 'fscanf', 
    'sscanf', 'gets', 'fgets', 'malloc', 'calloc', 'realloc', 'free', 
    'std::cout', 'cout', 'std::cerr', 'cerr', 'std::clog', 'clog',
    # Not Using '*' Heuristic since words can occur with or without std:: prefixation
    'exit', 'abort',
    'perror', 'system', 'setenv', 'unsetenv', 'atexit', 'signal', 'fopen', 'freopen', 'fclose', 'fflush',
    'fwrite', 'fread', 'fseek', 'ftell', 'rewind', 'remove', 'rename', 'tmpfile', 'tmpnam',
    'new[]', 'delete[]', 'std::terminate', 'std::abort', 'std::quick_exit'
}

# ------------------------------
# 2 CUDA keywords & intrinsics
# ------------------------------
# DOCS: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
# ------------------------------
cuda_storage_qualifiers = {
    '__global__', '__device__', '__host__', '__shared__', '__constant__', '__managed__', '__restrict__',
    '__launch_bounds__', '__cudart_builtin__'
}

cuda_synchronization = {
    '__threadfence_block', '__threadfence', '__syncthreads',
    '__threadfence_system', '__syncwarp', '__syncthreadfence', '__syncwarp_multi', '__sync_grid'
}

cuda_atomic = {
    'atomicAdd', 'atomicSub', 'atomicExch', 'atomicMin', 'atomicMax', 'atomicInc', 'atomicDec', 'atomicCAS',
    'atomicAnd', 'atomicOr', 'atomicXor',
    'atomicCompareAndSwap', 'atomicFetchAdd', 'atomicFetchSub', 'atomicFetchExch', 'atomicFetchMin',
    'atomicFetchMax', 'atomicFetchInc', 'atomicFetchDec', 'atomicFetchAnd', 'atomicFetchOr',
    'atomicFetchXor', 'atomicMax_block', 'atomicMin_block', 'atomicAdd_block', 'atomicExch_block',
    'atomicCAS_block', 'atomicAnd_block', 'atomicOr_block', 'atomicXor_block', 'atomicMax_system',
    'atomicMin_system', 'atomicAdd_system', 'atomicExch_system', 'atomicCAS_system',
    'atomicAnd_system', 'atomicOr_system', 'atomicXor_system', 'atomicMax_global', 'atomicMin_global',
    'atomicAdd_global', 'atomicExch_global', 'atomicCAS_global', 'atomicAnd_global', 'atomicOr_global',
    'atomicXor_global', 'atomicMax_shared', 'atomicMin_shared', 'atomicAdd_shared', 'atomicExch_shared',
    'atomicCAS_shared', 'atomicAnd_shared', 'atomicOr_shared', 'atomicXor_shared',
}

cuda_builtins = {
    'threadIdx', 'blockIdx', 'blockDim', 'gridDim',
    'warpSize', '__ldg', '__activemask', '__ballot_sync', '__shfl_sync', '__shfl_up_sync', '__shfl_down_sync',
    '__all_sync', '__any_sync', '__nvvm_reflect', '__builtin_shuffle'
}

cuda_types = {
    'dim3', 'float2', 'float3', 'float4', 'int2', 'int3', 'int4', 'uchar4', 'uint4',
    'cudaError_t', 'cudaStream_t', 'cudaEvent_t', 'texture', 'surface', 'size_t',
    'cudaArray_t', 'cudaChannelFormatDesc', 'cudaPitchedPtr'
}

cuda_side_effect_functions = {
    'cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaMemset', 'cudaMallocManaged', 'cudaMallocPitch',
    'cudaMemcpyAsync', 'cudaMemsetAsync', 'cudaDeviceSynchronize', 'cudaDeviceReset',
    'cudaStreamCreate', 'cudaStreamDestroy', 'cudaStreamSynchronize',
    'cudaEventCreate', 'cudaEventDestroy', 'cudaEventRecord', 'cudaEventSynchronize',
    'cudaGetLastError', 'cudaPeekAtLastError'
}

# ------------------------------
# 3 OpenCL keywords & types
# ------------------------------
# DOCS: https://ulhpc-tutorials.readthedocs.io/en/latest/gpu/opencl/
# ------------------------------
opencl_storage_qualifiers = {
    '__kernel', '__global', '__local', '__constant', '__private', '__attribute__((reqd_work_group_size))',
    '__attribute__((work_group_size_hint))'
}

opencl_functions = {
    'kernel', 'get_global_id', 'get_local_id', 'get_group_id', 'get_global_size', 'get_local_size', 'get_num_groups',
    'barrier', 'mem_fence', 'read_mem_fence', 'write_mem_fence',
    'get_work_dim', 'async_work_group_copy', 'wait_group_events', 'enqueue_kernel',
    'get_global_offset', 'get_sub_group_id'
}

opencl_memory_flags = {
    'CLK_LOCAL_MEM_FENCE', 'CLK_GLOBAL_MEM_FENCE',
    'CLK_IMAGE_MEM_FENCE', 'CLK_DOUBLE_MEM_FENCE'
}

opencl_types = {
    'cl_int', 'cl_uint', 'cl_float', 'cl_double', 'cl_char', 'cl_uchar', 'cl_short', 'cl_ushort',
    'cl_long', 'cl_ulong', 'cl_bool', 'cl_mem', 'float2', 'float4', 'int2', 'int4', 'size_t',
    'cl_half', 'cl_event', 'cl_sampler', 'cl_image_format', 'cl_half2', 'cl_half4'
}

opencl_side_effect_functions = {
    'clCreateBuffer', 'clReleaseMemObject', 'clEnqueueWriteBuffer', 'clEnqueueReadBuffer',
    'clCreateImage', 'clEnqueueNDRangeKernel', 'clFinish', 'clFlush', 'clSetKernelArg',
    'clEnqueueCopyBuffer', 'clEnqueueCopyImage'
}

# ------------------------------
# 4 Kokkos keywords & types
# ------------------------------
# DOCS: https://kokkos.org/kokkos-core-wiki/API/core/builtinreducers/MinLoc.html
# ------------------------------
kokkos_macros = {
    'KOKKOS_FUNCTION', 'KOKKOS_INLINE_FUNCTION', 'KOKKOS_LAMBDA',
    'KOKKOS_FORCEINLINE', 'KOKKOS_UNLIKELY'
}

kokkos_classes = {
    'Kokkos::View', 'Kokkos::DefaultExecutionSpace', 'Kokkos::DefaultHostExecutionSpace',
    'Kokkos::MemorySpace', 'Kokkos::LayoutLeft', 'Kokkos::LayoutRight',
    'Kokkos::DualView', 'Kokkos::Experimental', 'Kokkos::ViewHostMirror', 'Kokkos::Fencespace'
}

kokkos_parallel = {
    'Kokkos::parallel_for', 'Kokkos::parallel_reduce', 'Kokkos::parallel_scan',
    'Kokkos::TeamPolicy', 'Kokkos::RangePolicy', 'Kokkos::MDRangePolicy',
    'Kokkos::deep_copy', 'Kokkos::resize', 'Kokkos::create_mirror_view',
    'Kokkos::parallel_scan3d', 'Kokkos::parallel_for_reduce'
}

kokkos_side_effect_functions = {
    'Kokkos::initialize', 'Kokkos::finalize',
    'Kokkos::fence', 'Kokkos::hwloc_init'
}

# ------------------------------
# 5 OpenMP keywords & types
# ------------------------------
# DOCS: https://curc.readthedocs.io/en/latest/programming/OpenMP-C.html
# ------------------------------
openmp_pragmas = {
    'omp parallel', 'omp for', 'omp sections', 'omp single', 'omp master',
    'omp critical', 'omp atomic', 'omp barrier', 'omp task', 'omp taskwait',
    'omp parallel for', 'omp simd', 'omp reduction'
}

openmp_clauses = {
    'private', 'shared', 'firstprivate', 'lastprivate', 'reduction',
    'schedule', 'collapse', 'num_threads'
}

openmp_functions = {
    'omp_get_num_threads', 'omp_get_max_threads', 'omp_get_thread_num', 'omp_set_num_threads'
}

openmp_constants = {
    'omp_sched_static', 'omp_sched_dynamic', 'omp_sched_guided'
}

# ------------------------------
# 6 AdaptiveCPP (SYCL) keywords & types
# ------------------------------
# DOCS: https://adaptivecpp.github.io/AdaptiveCpp/extensions/
# ------------------------------
adaptivecpp_macros = {
    'SYCL_EXTERNAL', 'SYCL_UNROLL', 'SYCL_DEVICE_ONLY'
}

adaptivecpp_classes = {
    'sycl::queue', 'sycl::buffer', 'sycl::accessor', 'sycl::handler', 'sycl::event',
    'sycl::range', 'sycl::id', 'sycl::nd_range', 'sycl::nd_item', 'sycl::device', 'sycl::context',
    'sycl::program', 'sycl::platform', 'sycl::kernel', 'sycl::property', 'sycl::image', 'sycl::sampler'
}

adaptivecpp_parallel = {
    'sycl::parallel_for', 'sycl::parallel_for_work_group', 'sycl::single_task',
    'sycl::group_barrier', 'sycl::nd_item::barrier'
}

adaptivecpp_side_effect_functions = {
    'sycl::malloc_device', 'sycl::malloc_shared', 'sycl::malloc_host', 'sycl::free',
    'sycl::memcpy', 'sycl::memset', 'sycl::event::wait'
}

# ------------------------------
# 7 OpenACC keywords & types
# ------------------------------
# DOCS: https://enccs.github.io/OpenACC-CUDA-beginners/1.02_openacc-introduction/
# ------------------------------
openacc_clauses = {
    'copy', 'copyin', 'copyout', 'create', 'present', 'deviceptr',
    'num_gangs', 'num_workers', 'vector_length', 'collapse',
    'private', 'reduction', 'async', 'wait'
}

openacc_pragmas = {
    'acc parallel', 'acc kernels', 'acc loop', 'acc data',
    'acc host_data', 'acc enter data', 'acc exit data', 'acc update',
    'acc wait', 'acc routine'
}

openacc_functions = {
    'acc_get_device_type', 'acc_get_num_devices', 'acc_set_device_type',
    'acc_set_device_num', 'acc_malloc', 'acc_free', 'acc_memcpy_to_device',
    'acc_memcpy_from_device', 'acc_wait_all', 'acc_async_test'
}

openacc_constants = {
    'acc_device_nvidia', 'acc_device_radeon', 'acc_device_host', 'acc_device_default'
}

# ------------------------------
# 8 OpenGL / Vulkan keywords & types
# ------------------------------
# DOC: https://vkguide.dev/
# ------------------------------
opengl_vulkan_keywords = {
    # Vulkan C++ API classes (vulkan.hpp, vulkan_raii.hpp)
    'vk::Instance', 'vk::Device', 'vk::Queue', 'vk::CommandBuffer', 'vk::ShaderModule',
    'vk::Pipeline', 'vk::DescriptorSet', 'vk::Image', 'vk::Buffer', 'vk::PhysicalDevice',
    'vk::SurfaceKHR', 'vk::SwapchainKHR', 'vk::Semaphore', 'vk::Fence',
    'vk::RenderPass', 'vk::Framebuffer', 'vk::CommandPool', 'vk::Sampler',
    'vk::SubmitInfo', 'vk::PipelineLayout', 'vk::PipelineCache', 'vk::DescriptorSetLayout',
    'vk::Event', 'vk::ImageView', 'vk::BufferView', 'vk::DescriptorPool',
    'vk::ShaderStageFlagBits', 'vk::DescriptorType',

    # OpenGL symbols (gl.h / glad.h)
    'GL_TRIANGLES', 'GL_LINES', 'GL_POINTS', 'GL_COMPUTE_SHADER',
    'GL_VERTEX_SHADER', 'GL_FRAGMENT_SHADER', 'GL_GEOMETRY_SHADER', 'GL_UNIFORM_BUFFER',
    'GL_SHADER_STORAGE_BUFFER', 'GL_TEXTURE_2D', 'GL_ARRAY_BUFFER', 'GL_ELEMENT_ARRAY_BUFFER',
    'GL_FRAMEBUFFER', 'GL_RENDERBUFFER', 'GL_COLOR_ATTACHMENT0', 'GL_DEPTH_ATTACHMENT',
    'GL_DEPTH_TEST', 'GL_BLEND', 'GL_FLOAT', 'GL_UNSIGNED_INT', 'GL_STATIC_DRAW',
}

opengl_vulkan_functions = {
    # Vulkan C API
    'vkCreateInstance', 'vkDestroyInstance', 'vkEnumeratePhysicalDevices', 'vkCreateDevice',
    'vkGetDeviceQueue', 'vkCreateBuffer', 'vkCreateImage', 'vkAllocateMemory', 'vkBindBufferMemory',
    'vkBindImageMemory', 'vkCreateShaderModule', 'vkCreatePipelineLayout', 'vkCreateComputePipelines',
    'vkCmdDispatch', 'vkCmdBindPipeline', 'vkCmdBindDescriptorSets', 'vkCmdCopyBuffer',
    'vkQueueSubmit', 'vkQueueWaitIdle', 'vkDeviceWaitIdle', 'vkDestroyDevice', 'vkDestroyBuffer',
    'vkDestroyShaderModule', 'vkDestroyPipeline', 'vkFreeMemory', 'vkDestroyImage',
    'vkBeginCommandBuffer', 'vkEndCommandBuffer', 'vkResetCommandBuffer', '_vk_context',
    # OpenGL C API
    'glGenBuffers', 'glBindBuffer', 'glBufferData', 'glDeleteBuffers',
    'glCreateShader', 'glShaderSource', 'glCompileShader', 'glAttachShader',
    'glLinkProgram', 'glUseProgram', 'glDispatchCompute', 'glDrawArrays', 'glDrawElements',
    'glEnable', 'glDisable', 'glBindVertexArray', 'glGenVertexArrays', 'glDeleteVertexArrays',
    'glGetUniformLocation', 'glUniform1i', 'glUniformMatrix4fv', 'glGetAttribLocation',
    'glBindTexture', 'glActiveTexture', 'glTexImage2D', 'glTexParameteri',
    'glFramebufferTexture2D', 'glCheckFramebufferStatus',
}

opengl_vulkan_constants = {
    'VK_SUCCESS', 'VK_ERROR_OUT_OF_DEVICE_MEMORY', 'VK_ERROR_INITIALIZATION_FAILED',
    'VK_STRUCTURE_TYPE_APPLICATION_INFO', 'VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO',
    'VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO', 'VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO',
    'VK_QUEUE_COMPUTE_BIT', 'VK_PIPELINE_BIND_POINT_COMPUTE',
    'GL_COMPUTE_SHADER_BIT', 'GL_FRAGMENT_SHADER_BIT', 'GL_VERTEX_SHADER_BIT'
}

opengl_vulkan_macros = {
    'VK_NULL_HANDLE', 'VK_MAKE_VERSION', 'GL_CHECK_ERROR', 'GL_VERSION', 'VK_API_VERSION_1_2'
}
'''Edge Case: vk::PipelineCache matching along with vk::PipelineCacheInfro
TradeOff between Auto-Suffix Recognition and trying to recognise garbage constructs
'''

# ------------------------------
# 9 WebGPU keywords & types
# ------------------------------
# DOCS: https://docs.rs/wgpu/latest/wgpu/
# ------------------------------
webgpu_classes = {
    'wgpu::Device', 'wgpu::Queue', 'wgpu::CommandEncoder', 
    'wgpu::RenderPassEncoder', 'wgpu::Buffer', 'wgpu::Texture', 
    'wgpu::ShaderModule', 'wgpu::Pipeline', 'wgpu::BindGroup', 
    'wgpu::BindGroupLayout', 'wgpu::SwapChain', 'wgpu::CommandBuffer'
}

webgpu_functions = {
    'wgpuCreateInstance', 'wgpuDeviceCreateBuffer', 'wgpuDeviceCreateTexture',
    'wgpuDeviceCreateShaderModule', 'wgpuQueueSubmit', 'wgpuQueueWriteBuffer',
    'wgpuQueueCopyBufferToBuffer', 'wgpuQueueCopyBufferToTexture',
    'wgpuQueueCopyTextureToBuffer', 'wgpuCommandEncoderBeginRenderPass',
    'wgpuCommandEncoderFinish', 'wgpuSwapChainGetCurrentTextureView',
}

webgpu_constants = {
    'WGPU_BUFFER_USAGE_VERTEX', 'WGPU_BUFFER_USAGE_INDEX', 'WGPU_BUFFER_USAGE_UNIFORM',
    'WGPU_TEXTURE_USAGE_RENDER_ATTACHMENT', 'WGPU_TEXTURE_USAGE_COPY_SRC',
    'WGPU_TEXTURE_USAGE_COPY_DST', 'WGPU_TEXTURE_USAGE_SAMPLED'
}

webgpu_macros = {
    'WGPU_NULL_HANDLE', 'WGPU_CREATE_DEFAULT'
}

# ------------------------------
# 10 Boost.Compute keywords & types
# ------------------------------
# DOCS: https://www.boost.org/doc/libs/latest/libs/compute/doc/html/index.html
# ------------------------------
boost_macros = {
    'BOOST_COMPUTE_FUNCTION'
}

boost_classes = {
    'boost::compute::device', 'boost::compute::context', 'boost::compute::command_queue',
    'boost::compute::vector', 'boost::compute::system', 'boost::compute::function',
    'boost::compute::copy', 'boost::compute::transform'
}

boost_functions = {
    'BOOST_COMPUTE_FUNCTION',  # macro
    'boost::compute::copy', 'boost::compute::transform',
    'boost::compute::system::default_device',
    'boost::compute::command_queue::finish'
}

# No constants specifically for Boost.Compute

# ------------------------------
# 11 Metal keywords & types
# ------------------------------
# DOCS: https://developer.apple.com/documentation/metal/metal-sample-code-library
# ------------------------------
metal_storage = {
    # Metal storage qualifiers / keywords
    'device', 'thread', 'threadgroup', 'constant', 'kernel', 'sampler', 'texture',
    'thread_index_in_threadgroup', 'threadgroup_position_in_grid',
}

metal_functions = {
    # Side-effect functions
    'newBufferWithLength:options:', 'newTextureWithDescriptor:', 'commit', 'enqueue',
    'createComputePipelineStateWithFunction:error:', 'makeCommandQueue',
    'makeComputePipelineState', 'setBuffer:offset:atIndex:', 'dispatchThreads',
    'dispatchThreadgroups',
}

metal_classes = {
    # Metal classes / types
    'MTLDevice', 'MTLCommandQueue', 'MTLBuffer', 'MTLComputePipelineState',
    'MTLTexture', 'MTLCommandBuffer', 'MTLRenderPassDescriptor', 'MTLSamplerDescriptor',
    'MTLComputeCommandEncoder',
}

metal_constants = {
    # Constants / enums
    'MTLResourceCPUCacheModeDefaultCache', 'MTLResourceStorageModeShared',
    'MTLTextureType2D', 'MTLTextureUsageShaderRead'
}

# ------------------------------
# 12 Thrust keywords & types
# ------------------------------
# DOCS: https://nvidia.github.io/cccl/thrust/
# ------------------------------
thrust_classes = {
    'thrust::device_vector', 'thrust::host_vector', 'thrust::pair',
    'thrust::tuple', 'thrust::complex',
    'thrust::counting_iterator', 'thrust::constant_iterator',
    'thrust::transform_iterator', 'thrust::zip_iterator',
    'thrust::permutation_iterator', 'thrust::reverse_iterator',
    'thrust::sort_iterator',
}

thrust_functions = {
    'thrust::sort', 'thrust::transform', 'thrust::reduce', 'thrust::inclusive_scan',
    'thrust::exclusive_scan', 'thrust::copy', 'thrust::fill', 'thrust::count',
    'thrust::for_each', 'thrust::unique', 'thrust::remove', 'thrust::replace',
    'thrust::gather', 'thrust::scatter', 'thrust::sequence', 'thrust::merge',
    'thrust::inner_product', 'thrust::outer_product',
    'thrust::min_element', 'thrust::max_element',
}

thrust_macros = {
    'THRUST_DEVICE_SYSTEM_CUDA', 'THRUST_DEVICE_SYSTEM_OMP', 'THRUST_DEVICE_SYSTEM_TBB',
    'THRUST_HOST_SYSTEM_CPP', 'THRUST_HOST_SYSTEM_OMP', 'THRUST_HOST_SYSTEM_TBB',
    'THRUST_DEVICE_CODE', 'THRUST_HOST_CODE',
}

thrust_side_effect_functions = {
    'thrust::device_malloc', 'thrust::device_free',
    'thrust::system::cuda::malloc', 'thrust::system::cuda::free',
    'thrust::system::omp::malloc', 'thrust::system::omp::free',
    'thrust::system::tbb::malloc', 'thrust::system::tbb::free',
}

# ------------------------------
# 13 Merged Sets of Non-Operands by Language Extension
# ------------------------------
cpp_non_operands = cpp_control | cpp_types | cpp_modifiers | cpp_operators | cpp_side_effect_functions
cuda_non_operands = cuda_storage_qualifiers | cuda_synchronization | cuda_atomic | cuda_builtins | cuda_types | cuda_side_effect_functions
opencl_non_operands = opencl_storage_qualifiers | opencl_functions | opencl_memory_flags | opencl_types | opencl_side_effect_functions
kokkos_non_operands = kokkos_macros | kokkos_classes | kokkos_parallel | kokkos_side_effect_functions # Namespace functions must be kept
openmp_non_operands = openmp_pragmas | openmp_clauses | openmp_functions | openmp_constants
adaptivecpp_non_operands = adaptivecpp_macros | adaptivecpp_classes | adaptivecpp_parallel | adaptivecpp_side_effect_functions
openacc_non_operands = openacc_clauses | openacc_pragmas | openacc_functions | openacc_constants
opengl_vulkan_non_operands = opengl_vulkan_keywords | opengl_vulkan_functions | opengl_vulkan_constants | opengl_vulkan_macros
webgpu_non_operands = webgpu_classes | webgpu_functions | webgpu_constants | webgpu_macros
boost_non_operands = boost_macros | boost_classes | boost_functions
metal_non_operands = metal_storage | metal_functions | metal_classes | metal_constants
thrust_non_operands = thrust_classes | thrust_functions | thrust_macros | thrust_side_effect_functions

merged_non_operands = (
    cpp_non_operands
    | cuda_non_operands
    | opencl_non_operands
    | kokkos_non_operands
    | openmp_non_operands
    | adaptivecpp_non_operands
    | openacc_non_operands
    | opengl_vulkan_non_operands
    | webgpu_non_operands
    | boost_non_operands
    | metal_non_operands
    | thrust_non_operands
)

"""
EDGE CASE DOCUMENTATION:
IMPORTANT:
1) Not all function recognition is delegated to the suffixation patterns above. This is because:
- Some functions may be overloaded or templated, and hence may not follow a strict naming convention
- Some functions do not present a canonical suffix form 
- A base vocabulary minimizes false negatives
This may lead to rare double counting by counting on the invocation vk::QueueFlagBits
-> vk::Queue 
-> vk::QueueFlagBits
However this is an acceptable trade-off to ensure high recall of function recognition.
2) For pragma languages the '#pragma' keyword is treated as an operator itself since it behaves like a 
control directive.
Therefore '#pragma omp parallel for' is tokenized as:
-> '#pragma omp parallel for'  (#pragma operator)
-> 'omp parallel for' (actual function operator)

HANDLED: 
1) Some parallelizing frameworks contain communal keywords.
Therefore for the merging of non-operand sets with respect to CPP and the given parallelizing framework,
one needs to ensure that duplicates are allowed across parallelizing frameworks, just not within the
frameworks themselves -> so as to simplify the implementation.
Sets anyway don't allow duplicates, so this is inherently guaranteed. 
-> E.g. 
Both CUDA and OpenCL contain the keyword float2, hence float2 must be in both cuda_non_operands and opencl_non_operands,"
Structure:
-> Base CPP 
--> Parallizing Framework 1 (CUDA)    | Intersecting Overlap
--> Parallizing Framework 2 (OpenCL)  | may exist between all 3 frameworks
--> Parallizing Framework 3 (Kokkos)  | or between 2 given frameworks

NOT HANDLED:
1) Variables named with framework prefixations (However, this is bad code practice)
2) For the languages webgpu and metal, in code string kernels are handled as singular operands.
-> Only for OpenCL are the actual Kernel strings analyzed for operators and operands.
-> Only OpenCL kernel strings are analyzed for operators and operands because OpenCL embeds full C-like kernel programs
directly inside host language string literals. These strings contain valid program logic that corresponds to Halstead's 
definition of operators and operands.
-> In contrast, Metal (MSL) and WebGPU (WGSL) shader sources are separate languages with different syntactic rules that
cannot be reliably tokenized using C/C++ operator and operand patterns.
--> In Short OpenCL has a more C-like kernel language embedded in strings compared to the Metal and WebGPU 
semi-distinct shader languages.
"""


