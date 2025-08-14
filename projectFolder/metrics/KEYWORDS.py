## Assuming that Dattypes and Modifiers aren't considered operands and but hence be subtracted from the operand matches
## Subtracting non-operand keywords
# ------------------------------
# 1 Standard C++ keywords
# ------------------------------
cpp_control = {
    'if', 'else', 'switch', 'case', 'for', 'while', 'do', 'break', 'continue', 'return', 'goto', 'try', 'catch', 'throw'
}

cpp_types = {
    'int', 'float', 'double', 'char', 'void', 'short', 'long', 'signed', 'unsigned', 'bool',
    'wchar_t', 'char16_t', 'char32_t', 'class', 'struct', 'union', 'enum', 'namespace', 'typedef', 'using'
}

cpp_modifiers = {
    'public', 'private', 'protected', 'virtual', 'template', 'typename', 'static', 'const', 'volatile',
    'mutable', 'explicit', 'inline', 'friend', 'operator', 'this', 'extern', 'register', 'auto', 'thread_local',
    'static_assert', 'constexpr', 'decltype', 'export', 'import', 'module', 'requires', 'concept',
    'co_await', 'co_return', 'co_yield', 'asm', 'default', 'override', 'final', 'noexcept', 'nullptr_t', 'type'
}

cpp_operators = {
    '-', '+', '*', '/', '%', '=', '==', '!=', '<=', '>=', '<', '>', '&', '|', '^', '~', '!', '+=', '-=', '*=', '/=',
    '%=', '<<=', '>>=', '&=', '|=', '^=', '>>', '<<', '&&', '||', '++', '--', '->', '->*', '.', '::', '(', ')', '{', '}',
    '[', ']', ',', ':', ';', '#', '@', '...', '?'
}

cpp_side_effect_functions = {
    'printf', 'fprintf', 'sprintf', 'snprintf', 'puts', 'putchar',
    'scanf', 'fscanf', 'sscanf', 'gets', 'fgets',
    'malloc', 'calloc', 'realloc', 'free',
    'exit', 'abort'
}

# ------------------------------
# 2 CUDA keywords & intrinsics
# ------------------------------
cuda_storage_qualifiers = {
    '__global__', '__device__', '__host__', '__shared__', '__constant__', '__managed__', '__restrict__'
}

cuda_synchronization = {
    '__threadfence_block', '__threadfence', '__syncthreads'
}

cuda_atomic = {
    'atomicAdd', 'atomicSub', 'atomicExch', 'atomicMin', 'atomicMax', 'atomicInc', 'atomicDec', 'atomicCAS',
    'atomicAnd', 'atomicOr', 'atomicXor'
}

cuda_builtins = {
    'threadIdx', 'blockIdx', 'blockDim', 'gridDim'
}

cuda_types = {
    'dim3', 'float2', 'float3', 'float4', 'int2', 'int3', 'int4', 'uchar4', 'uint4',
    'cudaError_t', 'cudaStream_t', 'cudaEvent_t', 'texture', 'surface', 'size_t'
}

cuda_side_effect_functions = {
    'cudaMalloc', 'cudaFree', 'cudaMemcpy', 'cudaMemset', 'cudaMallocManaged', 'cudaMallocPitch'
}

# ------------------------------
# 3 OpenCL keywords & types
# ------------------------------
opencl_storage_qualifiers = {
    '__kernel', '__global', '__local', '__constant', '__private'
}

opencl_functions = {
    'kernel', 'get_global_id', 'get_local_id', 'get_group_id', 'get_global_size', 'get_local_size', 'get_num_groups',
    'barrier', 'mem_fence', 'read_mem_fence', 'write_mem_fence'
}

opencl_memory_flags = {
    'CLK_LOCAL_MEM_FENCE', 'CLK_GLOBAL_MEM_FENCE'
}

opencl_types = {
    'cl_int', 'cl_uint', 'cl_float', 'cl_double', 'cl_char', 'cl_uchar', 'cl_short', 'cl_ushort',
    'cl_long', 'cl_ulong', 'cl_bool', 'cl_mem', 'float2', 'float4', 'int2', 'int4', 'size_t'
}

opencl_side_effect_functions = {
    'clCreateBuffer', 'clReleaseMemObject', 'clEnqueueWriteBuffer', 'clEnqueueReadBuffer'
}

# ------------------------------
# 4 Kokkos keywords & types
# ------------------------------
kokkos_macros = {
    'KOKKOS_FUNCTION', 'KOKKOS_INLINE_FUNCTION', 'KOKKOS_LAMBDA'
}

kokkos_classes = {
    'Kokkos', 'Kokkos::View', 'Kokkos::DefaultExecutionSpace', 'Kokkos::DefaultHostExecutionSpace',
    'Kokkos::MemorySpace', 'Kokkos::LayoutLeft', 'Kokkos::LayoutRight'
}

kokkos_parallel = {
    'Kokkos::parallel_for', 'Kokkos::parallel_reduce', 'Kokkos::parallel_scan',
    'Kokkos::TeamPolicy', 'Kokkos::RangePolicy', 'Kokkos::MDRangePolicy'
}

kokkos_side_effect_functions = {
    'Kokkos::initialize', 'Kokkos::finalize'
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
