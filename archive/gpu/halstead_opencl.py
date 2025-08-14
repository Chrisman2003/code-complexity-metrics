import pyopencl as cl
import numpy as np

def gpu_compute_halstead(data):
    # Initialize OpenCL context and queue
    platforms = cl.get_platforms()
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    queue = cl.CommandQueue(context)

    # Prepare data for OpenCL kernel
    n1, n2, N1, N2 = data['n1'], data['n2'], data['N1'], data['N2']
    n1_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(n1, dtype=np.int32))
    n2_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(n2, dtype=np.int32))
    N1_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(N1, dtype=np.int32))
    N2_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(N2, dtype=np.int32))

    # Allocate output buffers
    volume_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=4)
    difficulty_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=4)
    effort_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=4)
    time_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=4)  
    
    # Define OpenCL kernel
    kernel_code = """
    __kernel void compute_halstead(__global const int *n1,
                                   __global const int *n2,
                                   __global const int *N1,
                                   __global const int *N2,
                                   __global float *volume,
                                   __global float *difficulty,
                                   __global float *effort,
                                   __global float *time) {
        int id = get_global_id(0);
        if (id == 0) {
            float v = (float)(n1[0] + n2[0]) * log2((float)(N1[0] + N2[0]));
            volume[0] = v;
            difficulty[0] = (float)n1[0] / 2.0f * (float)N2[0] / (float)N1[0];
            effort[0] = difficulty[0] * volume[0];
            time[0] = effort[0] / 18.0f; // Assuming 18 seconds per effort unit
        }
    }
    """

    program = cl.Program(context, kernel_code).build()

        # Launch kernel with a single work item
    program.compute_halstead(queue, (1,), None,
                             n1_buf, n2_buf, N1_buf, N2_buf,
                             volume_buf, difficulty_buf, effort_buf, time_buf)

    # Prepare numpy arrays to receive the results
    volume_np = np.empty(1, dtype=np.float32)
    difficulty_np = np.empty(1, dtype=np.float32)
    effort_np = np.empty(1, dtype=np.float32)
    time_np = np.empty(1, dtype=np.float32)

    # Read results from device to host
    cl.enqueue_copy(queue, volume_np, volume_buf)
    cl.enqueue_copy(queue, difficulty_np, difficulty_buf)
    cl.enqueue_copy(queue, effort_np, effort_buf)
    cl.enqueue_copy(queue, time_np, time_buf)

    # Wait for all operations to finish
    queue.finish()

    # Return results as a dictionary
    return {
        'volume': volume_np[0],
        'difficulty': difficulty_np[0],
        'effort': effort_np[0],
        'time': time_np[0]
    }