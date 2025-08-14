# ------------------------------
# 1 Standard C++ keywords + operators
# ------------------------------
cpp_keywords = {
    # Control structures
    'if', 'else', 'switch', 'case', 'for', 'while', 'do', 'break', 'continue', 'return', 'goto', 'try', 'catch', 'throw',
    'new', 'delete', 'sizeof', 'typeid', 'dynamic_cast', 'static_cast', 'reinterpret_cast', 'const_cast',
    'and', 'or', 'not', 'xor', 'bitand', 'bitor', 'compl', 'true', 'false', 'nullptr',
    # Type keywords
    'int', 'float', 'double', 'char', 'void', 'short', 'long', 'signed', 'unsigned', 'bool', 'wchar_t', 'char16_t', 'char32_t',
    'class', 'struct', 'union', 'enum', 'namespace', 'typedef', 'using',
    # C++ modifiers / specifiers
    'public', 'private', 'protected', 'virtual', 'template', 'typename', 'static', 'const', 'volatile', 'mutable',
    'explicit', 'inline', 'friend', 'operator', 'this', 'extern', 'register', 'auto', 'thread_local', 'static_assert',
    'constexpr', 'decltype', 'export', 'import', 'module', 'requires', 'concept', 'co_await', 'co_return', 'co_yield',
    'asm', 'default', 'override', 'final', 'noexcept', 'nullptr_t', 'type',
    # Operators (also include symbols)
    '-', '+', '*', '/', '%', '=', '==', '!=', '<=', '>=', '<', '>', '&', '|', '^', '~', '!', '+=', '-=', '*=', '/=',
    '%=', '<<=', '>>=', '&=', '|=', '^=', '>>', '<<', '&&', '||', '++', '--', '->', '->*', '.', '::', '(', ')', '{', '}',
    '[', ']', ',', ':', ';', '#', '@', '...', '?'
}

# ------------------------------
# 2 CUDA-specific keywords & intrinsics
# ------------------------------
cuda_keywords = {
    '__global__', '__device__', '__host__', '__shared__', '__constant__', '__managed__', '__restrict__',
    '__threadfence_block', '__threadfence', '__syncthreads',
    'atomicAdd', 'atomicSub', 'atomicExch', 'atomicMin', 'atomicMax', 'atomicInc', 'atomicDec', 'atomicCAS',
    'atomicAnd', 'atomicOr', 'atomicXor',
    # CUDA built-in variables (dim3 types)
    'threadIdx', 'blockIdx', 'blockDim', 'gridDim'
    # CUDA Types
    'dim3', 'float2', 'float3', 'float4', 'int2', 'int3', 'int4', 'uchar4', 'uint4',
    'cudaError_t', 'cudaStream_t', 'cudaEvent_t', 'texture', 'surface', 'size_t'
}

# ------------------------------
# 3️ OpenCL-specific keywords & types
# ------------------------------
opencl_keywords = {
    'kernel', '__kernel', '__global', '__local', '__constant', '__private',
    'get_global_id', 'get_local_id', 'get_group_id', 'get_global_size', 'get_local_size', 'get_num_groups',
    'barrier', 'mem_fence', 'read_mem_fence', 'write_mem_fence', 'CLK_LOCAL_MEM_FENCE', 'CLK_GLOBAL_MEM_FENCE'
    # OPENCL Types
    'cl_int', 'cl_uint', 'cl_float', 'cl_double', 'cl_char', 'cl_uchar', 'cl_short', 'cl_ushort', 'cl_long', 'cl_ulong',
    'cl_bool', 'cl_mem', 'float2', 'float4', 'int2', 'int4', 'size_t'
}

# ------------------------------
# 4 Kokkos-specific keywords & types
# ------------------------------
kokkos_keywords = {
    'Kokkos', 'KOKKOS_FUNCTION', 'KOKKOS_INLINE_FUNCTION', 'KOKKOS_LAMBDA',
    'Kokkos::parallel_for', 'Kokkos::parallel_reduce', 'Kokkos::parallel_scan',
    'Kokkos::TeamPolicy', 'Kokkos::RangePolicy', 'Kokkos::MDRangePolicy'
    # Kokkos Types
    'Kokkos::View', 'Kokkos::DefaultExecutionSpace', 'Kokkos::DefaultHostExecutionSpace', 'Kokkos::MemorySpace',
    'Kokkos::LayoutLeft', 'Kokkos::LayoutRight'
}

# ------------------------------
# 5 Merged sets for Halstead metric filtering
# ------------------------------
merged_keywords = cpp_keywords | cuda_keywords | opencl_keywords | kokkos_keywords
