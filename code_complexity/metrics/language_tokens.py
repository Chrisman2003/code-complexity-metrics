'''
Parallelizing Frameworks: 
1) CUDA
2) OpenCL
3) Kokkos
4) OpenMP 
5) AdaptiveCPP
6) OpenACC
7) OpenGlVulkan 
8) WebGPU
9) Boost
10) Metal
11) Thrust
[Future]

12) Slang: shading language
'''
# ------------------------------
# 1 Standard C++ keywords
# ------------------------------
cpp_control = {
    'if', 'else', 'switch', 'case', 'for', 'while', 'do', 'break', 'continue', 'return', 'goto',
    'try', 'catch', 'throw',
    # NEW
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
    # NEW
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
    # NEW
    'consteval', 'constinit', 'static_cast', 'const_cast',
    'false', 'true', 'nullptr', 'likely', 'unlikely', 'nodiscard', 'maybe_unused'
}

cpp_operators = {
    '-', '+', '*', '/', '%', '=', '==', '!=', '<=', '>=', '<', '>',
    '&', '|', '^', '~', '!', '+=', '-=', '*=', '/=', '%=', '<<=',
    '>>=', '&=', '|=', '^=', '>>', '<<', '&&', '||', '++', '--',
    '->', '->*', '.', '::', '(', ')', '{', '}',
    '[', ']', ',', ':', ';', '#', '@', '...', '?',
    # NEW
    'and', 'bitand', 'bitor', 'xor', 'compl', 'and_eq', 'xor_eq',
    '<=>', 'co_await_operator', 'operator<=>'
}

cpp_side_effect_functions = {
    'printf', 'fprintf', 'sprintf', 'snprintf', 'puts', 'putchar', 'scanf', 'fscanf', 
    'sscanf', 'gets', 'fgets', 'malloc', 'calloc', 'realloc', 'free', 
    'std::cout', 'cout', 'std::cerr', 'cerr', 'std::clog', 'clog',
    # Not Using '*' Heuristic since words can occur with or without std:: prefixation
    'exit', 'abort',
    # NEW
    'perror', 'system', 'setenv', 'unsetenv', 'atexit', 'signal', 'fopen', 'freopen', 'fclose', 'fflush',
    'fwrite', 'fread', 'fseek', 'ftell', 'rewind', 'remove', 'rename', 'tmpfile', 'tmpnam',
    'new[]', 'delete[]', 'std::terminate', 'std::abort', 'std::quick_exit'
}

# ------------------------------
# 2 CUDA keywords & intrinsics
# ------------------------------
cuda_storage_qualifiers = {
    '__global__', '__device__', '__host__', '__shared__', '__constant__', '__managed__', '__restrict__',
    # NEW
    '__launch_bounds__', '__cudart_builtin__'
}

cuda_synchronization = {
    '__threadfence_block', '__threadfence', '__syncthreads',
    # NEW
    '__threadfence_system', '__syncwarp', '__syncthreadfence', '__syncwarp_multi', '__sync_grid'
}

cuda_atomic = {
    'atomicAdd', 'atomicSub', 'atomicExch', 'atomicMin', 'atomicMax', 'atomicInc', 'atomicDec', 'atomicCAS',
    'atomicAnd', 'atomicOr', 'atomicXor',
    # NEW
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
    # NEW
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
    # NEW
    'cudaMemcpyAsync', 'cudaMemsetAsync', 'cudaDeviceSynchronize', 'cudaDeviceReset',
    'cudaStreamCreate', 'cudaStreamDestroy', 'cudaStreamSynchronize',
    'cudaEventCreate', 'cudaEventDestroy', 'cudaEventRecord', 'cudaEventSynchronize',
    'cudaGetLastError', 'cudaPeekAtLastError'
}

# ------------------------------
# 3 OpenCL keywords & types
# ------------------------------
opencl_storage_qualifiers = {
    '__kernel', '__global', '__local', '__constant', '__private', '__attribute__((reqd_work_group_size))',
    '__attribute__((work_group_size_hint))'
}

opencl_functions = {
    'kernel', 'get_global_id', 'get_local_id', 'get_group_id', 'get_global_size', 'get_local_size', 'get_num_groups',
    'barrier', 'mem_fence', 'read_mem_fence', 'write_mem_fence',
    # NEW
    'get_work_dim', 'async_work_group_copy', 'wait_group_events', 'enqueue_kernel',
    'get_global_offset', 'get_sub_group_id'
}

opencl_memory_flags = {
    'CLK_LOCAL_MEM_FENCE', 'CLK_GLOBAL_MEM_FENCE',
    # NEW
    'CLK_IMAGE_MEM_FENCE', 'CLK_DOUBLE_MEM_FENCE'
}

opencl_types = {
    'cl_int', 'cl_uint', 'cl_float', 'cl_double', 'cl_char', 'cl_uchar', 'cl_short', 'cl_ushort',
    'cl_long', 'cl_ulong', 'cl_bool', 'cl_mem', 'float2', 'float4', 'int2', 'int4', 'size_t',
    # NEW
    'cl_half', 'cl_event', 'cl_sampler', 'cl_image_format', 'cl_half2', 'cl_half4'
}

opencl_side_effect_functions = {
    'clCreateBuffer', 'clReleaseMemObject', 'clEnqueueWriteBuffer', 'clEnqueueReadBuffer',
    # NEW
    'clCreateImage', 'clEnqueueNDRangeKernel', 'clFinish', 'clFlush', 'clSetKernelArg',
    'clEnqueueCopyBuffer', 'clEnqueueCopyImage'
}

# ------------------------------
# 4 Kokkos keywords & types
# ------------------------------
kokkos_macros = {
    'KOKKOS_FUNCTION', 'KOKKOS_INLINE_FUNCTION', 'KOKKOS_LAMBDA',
    'KOKKOS_FORCEINLINE', 'KOKKOS_UNLIKELY'
}

kokkos_classes = {
    'Kokkos::View', 'Kokkos::DefaultExecutionSpace', 'Kokkos::DefaultHostExecutionSpace',
    'Kokkos::MemorySpace', 'Kokkos::LayoutLeft', 'Kokkos::LayoutRight',
    # NEW
    'Kokkos::DualView', 'Kokkos::Experimental', 'Kokkos::ViewHostMirror', 'Kokkos::Fencespace'
}

kokkos_parallel = {
    'Kokkos::parallel_for', 'Kokkos::parallel_reduce', 'Kokkos::parallel_scan',
    'Kokkos::TeamPolicy', 'Kokkos::RangePolicy', 'Kokkos::MDRangePolicy',
    # NEW
    'Kokkos::deep_copy', 'Kokkos::resize', 'Kokkos::create_mirror_view',
    'Kokkos::parallel_scan3d', 'Kokkos::parallel_for_reduce'
}

kokkos_side_effect_functions = {
    'Kokkos::initialize', 'Kokkos::finalize',
    # NEW
    'Kokkos::fence', 'Kokkos::hwloc_init'
}

# ------------------------------
# 5 OpenMP keywords & types
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

adaptivecpp_macros = {
    'SYCL_EXTERNAL', 'SYCL_UNROLL', 'SYCL_DEVICE_ONLY'
}

# ------------------------------
# 7 OpenACC keywords & types
# ------------------------------
openacc_pragmas = {
    'acc parallel', 'acc kernels', 'acc loop', 'acc data',
    'acc host_data', 'acc enter data', 'acc exit data', 'acc update',
    'acc wait', 'acc routine'
}

openacc_clauses = {
    'copy', 'copyin', 'copyout', 'create', 'present', 'deviceptr',
    'num_gangs', 'num_workers', 'vector_length', 'collapse',
    'private', 'reduction', 'async', 'wait'
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

# ------------------------------
# 9 WebGPU keywords & types
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

boost_macros = {
    'BOOST_COMPUTE_FUNCTION'
}
# No constants specifically for Boost.Compute

# ------------------------------
# 11 Metal keywords & types
# ------------------------------
metal_storage = {
    # Metal storage qualifiers / keywords
    'device', 'thread', 'threadgroup', 'constant', 'kernel', 'sampler', 'texture',
    'thread_index_in_threadgroup', 'threadgroup_position_in_grid',
}
metal_classes = {
    # Metal classes / types
    'MTLDevice', 'MTLCommandQueue', 'MTLBuffer', 'MTLComputePipelineState',
    'MTLTexture', 'MTLCommandBuffer', 'MTLRenderPassDescriptor', 'MTLSamplerDescriptor',
    'MTLComputeCommandEncoder',
}
metal_functions = {
    # Side-effect functions
    'newBufferWithLength:options:', 'newTextureWithDescriptor:', 'commit', 'enqueue',
    'createComputePipelineStateWithFunction:error:', 'makeCommandQueue',
    'makeComputePipelineState', 'setBuffer:offset:atIndex:', 'dispatchThreads',
    'dispatchThreadgroups',
}
metal_constants = {
    # Constants / enums
    'MTLResourceCPUCacheModeDefaultCache', 'MTLResourceStorageModeShared',
    'MTLTextureType2D', 'MTLTextureUsageShaderRead'
}

# ------------------------------
# 12 Thrust keywords & types
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
# 13 Slang shading language keywords & types
# ------------------------------
slang_keywords = {
    'interface', 'assoc', 'import', 'module', 'requires', 'where',
    'extension', 'this', 'inout', 'ref', 'let',

    # HLSL-compatible (Slang supports these)
    'cbuffer', 'tbuffer', 'SamplerState', 'SamplerComparisonState',
    'struct', 'class', 'enum',

    # Shader entry
    'compute', 'fragment', 'vertex', 'raygeneration', 'closesthit',
    'anyhit', 'miss', 'intersection', 'callable',

    # Layout qualifiers / parameter blocks
    'ConstantBuffer', 'ParameterBlock', 'ShaderRecordBuffer', 'groupshared',

    # Templates / generics
    '__generic', '__subscript', '__init', '__target_intrinsic',
}

slang_types = {
    # Scalars
    'int', 'uint', 'float', 'double', 'half', 'bool',

    # Vector types
    'float2', 'float3', 'float4',
    'int2', 'int3', 'int4',
    'uint2', 'uint3', 'uint4',
    'double2', 'double3', 'double4',
    'half2', 'half3', 'half4',

    # Matrices
    'float2x2', 'float3x3', 'float4x4',
    'double2x2', 'double3x3', 'double4x4',

    # GPU resource types
    'Texture1D', 'Texture2D', 'Texture3D', 'TextureCube',
    'Texture1DArray', 'Texture2DArray', 'TextureCubeArray',
    'RWTexture1D', 'RWTexture2D', 'RWTexture3D',

    'Buffer', 'RWBuffer', 'StructuredBuffer', 'RWStructuredBuffer',
    'ByteAddressBuffer', 'RWByteAddressBuffer',

    # Acceleration structures
    'RaytracingAccelerationStructure',

    # Samplers
    'SamplerState', 'SamplerComparisonState',

    # Special Slang constructs
    'ParameterBlock', 'ConstantBuffer', 'ShaderStruct', 'Payload',
}

slang_modifiers = {
    'static', 'const', 'nointerpolation', 'precise',
    'uniform', 'inline', 'unroll', 'loop', 'branch',
    'flatten', 'forceinline', 'numthreads',

    # Slang-specific
    '__exported', '__transparent', '__specialize',
    '__subscript', '__subscriptMutable', '__generic',
    '__intrinsic', '__builtin', '__target_intrinsic',
}

slang_semantics = {
    'SV_Position', 'SV_Target', 'SV_DispatchThreadID', 'SV_GroupID',
    'SV_GroupThreadID', 'SV_GroupIndex', 'SV_InstanceID', 'SV_PrimitiveID',
    'SV_SampleIndex', 'SV_VertexID', 'SV_IsFrontFace',

    # Ray tracing
    'SV_RayDirection', 'SV_RayTMin', 'SV_RayTCurrent',
    'SV_RayPayload', 'SV_GeometryIndex', 'SV_HitKind',
}

slang_intrinsics = {
    # Math
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'exp', 'log', 'sqrt', 'abs', 'floor', 'ceil',
    'clamp', 'lerp', 'normalize', 'cross', 'dot',
    'min', 'max', 'pow', 'rcp', 'saturate',

    # Matrix transforms
    'mul', 'transpose', 'determinant',

    # Wave ops
    'WaveActiveSum', 'WaveActiveMin', 'WaveActiveMax',
    'WavePrefixSum', 'WaveReadLaneAt', 'WaveGetLaneIndex',

    # Texture operations
    'Sample', 'SampleLevel', 'Load', 'GatherRed', 'Gather',

    # Ray tracing intrinsics (DXR / Vulkan RT)
    'TraceRay', 'ReportHit', 'IgnoreHit', 'AcceptHitAndEndSearch',
}

slang_operators = {
    '+', '-', '*', '/', '%', '=', '==', '!=', '<=', '>=', '<', '>',
    '&&', '||', '!', '~', '&', '|', '^',
    '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^=',
    '<<', '>>', '++', '--', '->', '.', '::', ',', ':', ';', '?',
    '(', ')', '{', '}', '[', ']',
    # Slang generics
    '<', '>',  # (parsed separately from operators, but included)
}

slang_side_effect_functions = {
    # Resource writes
    'InterlockedAdd', 'InterlockedMin', 'InterlockedMax',
    'InterlockedExchange', 'InterlockedCompareExchange',

    # UAV / RW structured buffer writes
    'RWTexture2D.Store', 'RWTexture3D.Store',
    'RWStructuredBuffer.Append', 'RWStructuredBuffer.Consume',

    # Dispatch & tracing side effects
    'TraceRay', 'ReportHit', 'IgnoreHit',

    # Debug
    'printf', 'abort',
}



# ------------------------------
# 14 Merged Subsets per type for Component Analysis of Halstead metrics
# ------------------------------
# TODO
merged_control = cpp_control
merged_types = cpp_types | cuda_types | opencl_types | kokkos_classes
merged_modifiers = cpp_modifiers | cuda_storage_qualifiers | opencl_storage_qualifiers | kokkos_macros
merged_operators = cpp_operators | cpp_side_effect_functions | cuda_side_effect_functions | opencl_side_effect_functions | kokkos_side_effect_functions
merged_functions = cpp_side_effect_functions | cuda_side_effect_functions | opencl_side_effect_functions | kokkos_side_effect_functions
merged_parallel = cuda_atomic | cuda_synchronization | opencl_functions | kokkos_parallel
# TODO

# ------------------------------
# 15 Merged Sets of Non-Operands by Language Extension
# ------------------------------
cpp_non_operands = cpp_control | cpp_types | cpp_modifiers | cpp_operators | cpp_side_effect_functions
cuda_non_operands = cuda_storage_qualifiers | cuda_synchronization | cuda_atomic | cuda_builtins | cuda_types | cuda_side_effect_functions
opencl_non_operands = opencl_storage_qualifiers | opencl_functions | opencl_memory_flags | opencl_types | opencl_side_effect_functions
kokkos_non_operands = kokkos_macros | kokkos_classes | kokkos_parallel | kokkos_side_effect_functions
openmp_non_operands = openmp_pragmas | openmp_clauses | openmp_functions | openmp_constants
adaptivecpp_non_operands = adaptivecpp_classes | adaptivecpp_parallel | adaptivecpp_side_effect_functions | adaptivecpp_macros
openacc_non_operands = openacc_pragmas | openacc_clauses | openacc_functions | openacc_constants
opengl_vulkan_non_operands = opengl_vulkan_keywords | opengl_vulkan_functions | opengl_vulkan_constants | opengl_vulkan_macros
webgpu_non_operands = webgpu_classes | webgpu_functions | webgpu_constants | webgpu_macros
boost_non_operands = boost_classes | boost_functions | boost_macros
metal_non_operands = metal_storage | metal_classes | metal_functions | metal_constants
thrust_non_operands = thrust_classes | thrust_functions | thrust_macros | thrust_side_effect_functions
slang_non_operands = slang_keywords | slang_types | slang_modifiers | slang_semantics | slang_intrinsics | slang_operators | slang_side_effect_functions

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
    | slang_non_operands
)

'''
CRUCIAL Edge Case:
Some parallelizing frameworks contain communal keywords.
Therefore for the merging of non-operands sets with respect to CPP and the given parallelizing framework,
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
'''

"Edge Case: remove libraries before analysis"


'''
Libraries:
1) CUDA: #include <cuda_runtime.hpp>
2) KOKKOS: #include <Kokkos_Core.hpp>
3) VULKAN/OPENGL: #include <vulkan/vulkan.hpp>
   #include <vulkan/vulkan_raii.hpp>
   #include <glm/glm.hpp> 
4) ADAPTIVECPP: #include <CL/sycl.hpp>
   #include <sycl/sycl.hpp>
5) THRUST: #include <thrust/device_vector.h>
   #include <thrust/reduce.h>
6) OPENACC: #include <openacc.h>
7) OPENCL: #include <CL/cl.hpp>
   #include "opencl_eval.hpp"
   #include "opencl_init.hpp"
   #include "opencl_sum.hpp"
8) OPENMP: #include "omp.h"
9) SLANG: #include "shader/eval.hpp"
   #include "shader/eval.cuh"
10) WEBGPU: #include <wgpu/wgpu.h>
11) BOOST: #include "boost/compute.hpp"
12) METAL: #include <Metal/Metal.hpp>
'''