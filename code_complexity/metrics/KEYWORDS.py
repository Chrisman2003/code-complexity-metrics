'''
Parallelizing Frameworks: 
1) CUDA
2) OpenCL
3) Kokkos
[Future]
4) AdaptiveCPP
5) OpenMP 

6) OpenACC
7) OpenGlVulkan 
8) Slang 
9) WebGPU
10) Boost
11) Metal
12) Thrust
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
}

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
    '[', ']', ',', ':', ';', '#', '@', '...', '?'
    # NEW
    'and', 'bitand', 'bitor', 'xor', 'compl', 'and_eq', 'xor_eq',
    '<=>', 'co_await_operator', 'operator<=>'
}

cpp_side_effect_functions = {
    'printf', 'fprintf', 'sprintf', 'snprintf', 'puts', 'putchar', 'scanf', 'fscanf', 
    'sscanf', 'gets', 'fgets', 'malloc', 'calloc', 'realloc', 'free', 
    'std::cout', 'cout', 'std::cerr', 'cerr', 'std::clog', 'clog' 
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
# 5 Merged Subsets per type for Component Analysis of Halstead metrics
# ------------------------------
merged_control = cpp_control
merged_types = cpp_types | cuda_types | opencl_types | kokkos_classes
merged_modifiers = cpp_modifiers | cuda_storage_qualifiers | opencl_storage_qualifiers | kokkos_macros
merged_operators = cpp_operators | cpp_side_effect_functions | cuda_side_effect_functions | opencl_side_effect_functions | kokkos_side_effect_functions
merged_functions = cpp_side_effect_functions | cuda_side_effect_functions | opencl_side_effect_functions | kokkos_side_effect_functions
merged_parallel = cuda_atomic | cuda_synchronization | opencl_functions | kokkos_parallel

# ------------------------------
# 6 Merged Sets by Language Extension
# ------------------------------
cpp_non_operands = cpp_control | cpp_types | cpp_modifiers | cpp_operators | cpp_side_effect_functions
cuda_non_operands = cuda_storage_qualifiers | cuda_synchronization | cuda_atomic | cuda_builtins | cuda_types | cuda_side_effect_functions
opencl_non_operands = opencl_storage_qualifiers | opencl_functions | opencl_memory_flags | opencl_types | opencl_side_effect_functions
kokkos_non_operands = kokkos_macros | kokkos_classes | kokkos_parallel | kokkos_side_effect_functions

merged_non_operands = cpp_non_operands | cuda_non_operands | opencl_non_operands | kokkos_non_operands

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