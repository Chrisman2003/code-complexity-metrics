// complex_boost_compute.cpp
// Compile with: g++ complex_boost_compute.cpp -lOpenCL -lboost_system -std=c++17
// (exact link flags may vary by platform)

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cassert>

#include <boost/compute.hpp>

int main() {
    // -------------------------
    // Setup device, context, queue
    // -------------------------
    boost::compute::device device = boost::compute::system::default_device();
    std::cout << "Using device: " << device.name() << std::endl;

    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    // -------------------------
    // Problem size and host data
    // -------------------------
    constexpr size_t N = 1 << 16; // 65536 points (adjust for memory/time)
    std::vector<float> host_points; // packed x,y,z,w for each point -> size = 4*N
    host_points.resize(4 * N);

    // generate random points in a cube [-10,10]^3
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (size_t i = 0; i < N; ++i) {
        host_points[4*i + 0] = dist(rng); // x
        host_points[4*i + 1] = dist(rng); // y
        host_points[4*i + 2] = dist(rng); // z
        host_points[4*i + 3] = 1.0f;      // w (unused)
    }

    // Choose a query point in space
    const float qx = 0.5f, qy = -1.2f, qz = 3.3f;
    const float eps = 1e-6f;

    // -------------------------
    // Device buffers
    // -------------------------
    // points_buffer: float4 array (x,y,z,w)
    boost::compute::vector<float> d_points(4 * N, context);
    // potential contributions per point
    boost::compute::vector<float> d_potentials(N, context);
    // distances for sorting
    boost::compute::vector<float> d_dist2(N, context);
    // indices for stable sort_by_key
    boost::compute::vector<uint32_t> d_indices(N, context);

    // copy host points to device
    auto t0 = std::chrono::high_resolution_clock::now();
    boost::compute::copy(host_points.begin(), host_points.end(), d_points.begin(), queue);
    queue.finish();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Uploaded " << N << " points in "
              << std::chrono::duration<double>(t1 - t0).count() << " s\n";

    // initialize indices 0..N-1 on device
    boost::compute::iota(d_indices.begin(), d_indices.end(), static_cast<uint32_t>(0), queue);

    // -------------------------
    // Build custom OpenCL kernel: compute potential and squared distance
    // -------------------------
    // Kernel source: reads packed points (float4 per point), writes 1/r to potentials and squared distance to dist2
    const std::string src = R"(
    __kernel void eval_potential(__global const float4* points,
                                 __global float* potentials,
                                 __global float* dist2,
                                 const float qx,
                                 const float qy,
                                 const float qz,
                                 const float eps) {
        const uint gid = get_global_id(0);
        float4 p = points[gid];
        float dx = p.x - qx;
        float dy = p.y - qy;
        float dz = p.z - qz;
        float r2 = dx*dx + dy*dy + dz*dz + eps;
        float r = sqrt(r2);
        potentials[gid] = 1.0f / r;
        dist2[gid] = r2;
    }
    )";

    boost::compute::program program = boost::compute::program::create_with_source(src, context);
    try {
        program.build({device});
    } catch (boost::compute::opencl_error &e) {
        std::cerr << "Build log:\n" << program.build_log() << "\n";
        throw;
    }

    boost::compute::kernel kernel(program, "eval_potential");

    // set kernel args (we'll set scalar args later each dispatch)
    kernel.set_arg(0, d_points.get_buffer());     // __global float4* points
    kernel.set_arg(1, d_potentials.get_buffer()); // __global float* potentials
    kernel.set_arg(2, d_dist2.get_buffer());      // __global float* dist2

    // -------------------------
    // Launch kernel: compute potentials and squared distances
    // -------------------------
    const size_t work_group_size = 256;
    const size_t global_size = ((N + work_group_size - 1) / work_group_size) * work_group_size;

    // set scalar args
    kernel.set_arg(3, qx);
    kernel.set_arg(4, qy);
    kernel.set_arg(5, qz);
    kernel.set_arg(6, eps);

    auto t2 = std::chrono::high_resolution_clock::now();
    queue.enqueue_nd_range_kernel(kernel, 1, NULL, &global_size, &work_group_size);
    queue.finish();
    auto t3 = std::chrono::high_resolution_clock::now();

    std::cout << "Kernel eval_potential executed in "
              << std::chrono::duration<double>(t3 - t2).count() << " s\n";

    // -------------------------
    // Reduce: compute total potential (sum of potentials)
    // -------------------------
    auto t4 = std::chrono::high_resolution_clock::now();
    float total_potential = boost::compute::reduce(d_potentials.begin(), d_potentials.end(), 0.0f, boost::compute::plus<float>(), queue);
    queue.finish();
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "Reduced total potential (device) = " << total_potential
              << " computed in " << std::chrono::duration<double>(t5 - t4).count() << " s\n";

    // -------------------------
    // Sort by distance squared using sort_by_key (dist2 keys, indices values)
    // -------------------------
    auto t6 = std::chrono::high_resolution_clock::now();
    boost::compute::sort_by_key(d_dist2.begin(), d_dist2.end(), d_indices.begin(), queue);
    queue.finish();
    auto t7 = std::chrono::high_resolution_clock::now();
    std::cout << "Sorted " << N << " distances in "
              << std::chrono::duration<double>(t7 - t6).count() << " s\n";

    // -------------------------
    // Copy top-K nearest points back to host (sample K=16)
    // -------------------------
    const size_t K = 16;
    std::vector<uint32_t> host_top_indices(K);
    boost::compute::copy(d_indices.begin(), d_indices.begin() + K, host_top_indices.begin(), queue);

    std::vector<float> host_top_points(4 * K);
    for (size_t i = 0; i < K; ++i) {
        size_t idx = host_top_indices[i];
        // copy single float4 from device to host (one element). For efficiency we could gather, but do per-element read for clarity.
        boost::compute::vector<float> tmp4(4, context);
        boost::compute::copy(d_points.begin() + 4*idx, d_points.begin() + 4*idx + 4, tmp4.begin(), queue);
        boost::compute::copy(tmp4.begin(), tmp4.end(), host_top_points.begin() + 4*i, queue);
    }
    queue.finish();

    std::cout << "Top-" << K << " nearest points (x,y,z) and distances:\n";
    // fetch corresponding distances too
    std::vector<float> host_top_dist2(K);
    boost::compute::copy(d_dist2.begin(), d_dist2.begin() + K, host_top_dist2.begin(), queue);
    for (size_t i = 0; i < K; ++i) {
        float x = host_top_points[4*i + 0];
        float y = host_top_points[4*i + 1];
        float z = host_top_points[4*i + 2];
        float r2 = host_top_dist2[i];
        std::cout << std::setw(2) << i << ": idx=" << host_top_indices[i]
                  << " pos=(" << x << "," << y << "," << z << ") r=" << std::sqrt(r2) << "\n";
    }

    // -------------------------
    // CPU verification (optional, small N)
    // -------------------------
    if (N <= 100000) {
        // compute on host for spot-check
        std::vector<float> host_potentials(N);
        for (size_t i = 0; i < N; ++i) {
            float dx = host_points[4*i+0] - qx;
            float dy = host_points[4*i+1] - qy;
            float dz = host_points[4*i+2] - qz;
            float r = std::sqrt(dx*dx + dy*dy + dz*dz + eps);
            host_potentials[i] = 1.0f / r;
        }
        double host_total = 0.0;
        for (auto v : host_potentials) host_total += v;
        std::cout << "Host total potential (reference) = " << host_total << "\n";
        std::cout << "Relative GPU/CPU difference = " << std::abs(host_total - total_potential) / std::max(1.0, std::abs(host_total)) << "\n";
    }

    std::cout << "Done.\n";
    return 0;
}
