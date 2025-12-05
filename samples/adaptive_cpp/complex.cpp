#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <limits>

namespace sycl = cl::sycl;

// Utility: pretty chrono timer
struct ScopedTimer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    std::string name;
    ScopedTimer(const std::string& n) : start(clock::now()), name(n) {}
    double stop_seconds() {
        auto diff = std::chrono::duration<double>(clock::now() - start).count();
        return diff;
    }
    ~ScopedTimer() {}
};

// Basic matrix container helpers (row-major)
inline size_t idx(size_t r, size_t c, size_t cols) { return r * cols + c; }

// Subgroup-assisted reduce: sum within a subgroup using shuffle_down
template <typename T>
T subgroup_reduce_sum(sycl::sub_group sg, T value) {
    // assume power-of-two subgroup size or device shuffle supports arbitrary
    for (auto offset = sg.get_local_range()[0] / 2; offset > 0; offset >>= 1) {
        value += sg.shuffle_down(value, offset);
    }
    return value;
}

int main(int argc, char** argv) {
    try {
        // Select device: prefer GPU, fallback to CPU
        sycl::device dev;
        try {
            dev = sycl::device(sycl::gpu_selector{});
        } catch (...) {
            std::cout << "GPU not found - falling back to default selector\n";
            dev = sycl::device(sycl::default_selector{});
        }
        sycl::queue q(dev, sycl::property::queue::in_order{});

        std::cout << "Running on: "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << "\n";

        // Problem size parameters - keep moderate for demo but large enough
        const size_t M = 512; // rows of A and C
        const size_t K = 512; // cols of A, rows of B
        const size_t N = 512; // cols of B and C

        // Autotune candidates for tile size (must divide local range nicely)
        std::vector<int> tileCandidates = {8, 16, 32}; // try a few tile sizes
        int bestTile = tileCandidates[0];
        double bestTime = std::numeric_limits<double>::infinity();

        // Allocate USM shared arrays for easy host access (A,B) and buffer for C to demo hybrid
        float* A = sycl::malloc_shared<float>(M * K, q);
        float* B = sycl::malloc_shared<float>(K * N, q);
        // We'll use a buffer for C to demonstrate accessor-based kernel writing
        std::vector<float> C_host(M * N, 0.0f);
        sycl::buffer<float, 2> bufC((float*)C_host.data(), sycl::range<2>(M, N));

        // Initialize A and B with pseudo-random values
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < M * K; ++i) A[i] = dist(rng);
        for (size_t i = 0; i < K * N; ++i) B[i] = dist(rng);

        // Warmup/verification arrays for CPU reference
        std::vector<float> C_ref(M * N, 0.0f);

        auto cpu_gemm = [&](int tile) {
            // naive CPU gemm (for verification) - single-threaded (slow but simple)
            for (size_t r = 0; r < M; ++r) {
                for (size_t c = 0; c < N; ++c) {
                    float s = 0.0f;
                    for (size_t k = 0; k < K; ++k) {
                        s += A[idx(r, k, K)] * B[idx(k, c, N)];
                    }
                    C_ref[idx(r, c, N)] = s;
                }
            }
        };

        std::cout << "Doing a CPU reference GEMM for verification...\n";
        {
            ScopedTimer t("CPU GEMM");
            cpu_gemm(bestTile);
            double tsec = t.stop_seconds();
            std::cout << "CPU GEMM took " << std::fixed << std::setprecision(3) << tsec << " s\n";
        }

        // Autotune loop: try different tile sizes and measure kernel time
        for (int tile : tileCandidates) {
            // Heuristics: tile must be <= both N and M and K generally
            if (tile <= 0 || tile > static_cast<int>(std::min(M, N))) continue;

            // Reset C_host via buffer
            {
                auto acc = bufC.get_access<sycl::access::mode::discard_write>();
                for (size_t r = 0; r < M; ++r)
                    for (size_t c = 0; c < N; ++c) acc[r][c] = 0.0f;
            }

            // Build ND-range based on tile
            sycl::range<2> local(tile, tile);
            sycl::range<2> global( (N + tile - 1) / tile * tile, (M + tile - 1) / tile * tile );
            sycl::nd_range<2> ndr(global, local);

            // Time a single kernel run for this tile size
            double kernel_time = 0.0;
            {
                ScopedTimer t("GEMM kernel run");
                // Submit kernel
                q.submit([&](sycl::handler& h) {
                    // local tile storage (shared memory)
                    sycl::local_accessor<float, 2> tileA(sycl::range<2>(tile, tile), h);
                    sycl::local_accessor<float, 2> tileB(sycl::range<2>(tile, tile), h);

                    auto accC = bufC.get_access<sycl::access::mode::read_write>(h);

                    h.parallel_for<class TiledGEMM>(ndr, [=](sycl::nd_item<2> it) {
                        size_t global_col = it.get_global_id(0);
                        size_t global_row = it.get_global_id(1);

                        size_t local_col = it.get_local_id(0);
                        size_t local_row = it.get_local_id(1);

                        float acc = 0.0f;

                        auto sg = it.get_sub_group();

                        // iterate tiles along K
                        for (size_t t0 = 0; t0 < K; t0 += tile) {
                            // Load A and B pieces into local memory (if in range)
                            size_t a_row = global_row;
                            size_t a_col = t0 + local_col;
                            if (a_row < M && a_col < K)
                                tileA[local_row][local_col] = A[idx(a_row, a_col, K)];
                            else
                                tileA[local_row][local_col] = 0.0f;

                            size_t b_row = t0 + local_row;
                            size_t b_col = global_col;
                            if (b_row < K && b_col < N)
                                tileB[local_row][local_col] = B[idx(b_row, b_col, N)];
                            else
                                tileB[local_row][local_col] = 0.0f;

                            it.barrier(sycl::access::fence_space::local_space);

                            // compute partial products
                            // To use subgroup shuffles we compute product per lane and reduce across subgroup
                            for (int kk = 0; kk < tile; ++kk) {
                                float p = tileA[local_row][kk] * tileB[kk][local_col];
                                // reduce within subgroup
                                float r = subgroup_reduce_sum(sg, p);
                                // Each subgroup lane will get the full subgroup-sum; we divide by subgroup size and let one lane accumulate
                                // But to keep it simple and robust across devices we divide and add per lane:
                                acc += r / static_cast<float>(sg.get_local_range()[0]);
                            }

                            it.barrier(sycl::access::fence_space::local_space);
                        }

                        // write result if within bounds
                        if (global_row < M && global_col < N) {
                            accC[global_row][global_col] = acc;
                        }
                    });
                });

                q.wait();
                kernel_time = t.stop_seconds();
                std::cout << "Tile=" << tile << " kernel time: " << kernel_time << " s\n";
            }

            if (kernel_time < bestTime) {
                bestTime = kernel_time;
                bestTile = tile;
            }
        } // end autotune loop

        std::cout << "Autotune selected tile = " << bestTile
                  << " (best time = " << bestTime << " s)\n";

        // Now run a multi-stage pipeline:
        // 1) Transpose B into Bt to improve memory access patterns for GEMM (USM)
        // 2) Run final GEMM with the best tile, reading Bt
        // 3) Reduce checksum on device using a reduction kernel (buffers + atomic)
        // 4) Copy C to host and verify against CPU reference

        // 1) Transpose B -> Bt (USM allocate)
        float* Bt = sycl::malloc_shared<float>(N * K, q); // transpose dims: N x K (so Bt[c,k] = B[k,c])
        {
            sycl::range<2> globalT( (N + 15)/16 * 16, (K + 15)/16 * 16 );
            sycl::range<2> localT(16, 16);

            q.submit([&](sycl::handler& h) {
                h.parallel_for<class TransposeB>(sycl::nd_range<2>(globalT, localT),
                [=](sycl::nd_item<2> it) {
                    size_t col = it.get_global_id(0);
                    size_t row = it.get_global_id(1);
                    if (row < K && col < N) {
                        Bt[idx(col, row, K)] = B[idx(row, col, N)];
                    }
                });
            }).wait();
        }

        // 2) Final tiled GEMM reading A and Bt (Bt accessed as [col,row] -> Bt[col*K + row])
        {
            // Reset C buffer
            {
                auto acc = bufC.get_access<sycl::access::mode::discard_write>();
                for (size_t r = 0; r < M; ++r)
                    for (size_t c = 0; c < N; ++c) acc[r][c] = 0.0f;
            }

            int tile = bestTile;
            sycl::range<2> local(tile, tile);
            sycl::range<2> global( (N + tile - 1) / tile * tile, (M + tile - 1) / tile * tile );
            sycl::nd_range<2> ndr(global, local);

            ScopedTimer t("Final GEMM");
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> tileA(sycl::range<2>(tile, tile), h);
                sycl::local_accessor<float, 2> tileB(sycl::range<2>(tile, tile), h);
                auto accC = bufC.get_access<sycl::access::mode::read_write>(h);

                h.parallel_for<class FinalGEMM>(ndr, [=](sycl::nd_item<2> it) {
                    size_t global_col = it.get_global_id(0);
                    size_t global_row = it.get_global_id(1);

                    size_t local_col = it.get_local_id(0);
                    size_t local_row = it.get_local_id(1);

                    float acc = 0.0f;
                    auto sg = it.get_sub_group();

                    for (size_t t0 = 0; t0 < K; t0 += tile) {
                        // load A tile
                        size_t a_row = global_row;
                        size_t a_col = t0 + local_col;
                        if (a_row < M && a_col < K)
                            tileA[local_row][local_col] = A[idx(a_row, a_col, K)];
                        else
                            tileA[local_row][local_col] = 0.0f;

                        // load Bt tile (note Bt is transposed: Bt[col,row] = B[row,col])
                        size_t bt_row = global_col;
                        size_t bt_col = t0 + local_row;
                        if (bt_row < N && bt_col < K)
                            tileB[local_row][local_col] = Bt[idx(bt_row, bt_col, K)];
                        else
                            tileB[local_row][local_col] = 0.0f;

                        it.barrier(sycl::access::fence_space::local_space);

                        for (int kk = 0; kk < tile; ++kk) {
                            float p = tileA[local_row][kk] * tileB[kk][local_col];
                            float r = subgroup_reduce_sum(sg, p);
                            acc += r / static_cast<float>(sg.get_local_range()[0]);
                        }

                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    if (global_row < M && global_col < N) {
                        accC[global_row][global_col] = acc;
                    }
                });
            });

            q.wait();
            double final_time = t.stop_seconds();
            std::cout << "Final GEMM elapsed: " << final_time << " s\n";
        }

        // 3) Device reduction for checksum: we'll read from bufC via accessor and do a parallel sum into a USM scalar
        float* device_sum = sycl::malloc_shared<float>(1, q);
        device_sum[0] = 0.0f;
        {
            // We'll use a reduction-like kernel by assigning one work-item per row and reduce across columns
            sycl::range<1> globalR(M);
            sycl::range<1> localR(64); // work-group size
            sycl::nd_range<1> ndr(globalR, localR);

            q.submit([&](sycl::handler& h) {
                auto accC = bufC.get_access<sycl::access::mode::read>(h);
                h.parallel_for<class Checksum>(ndr, [=](sycl::nd_item<1> it) {
                    size_t r = it.get_global_id(0);
                    float row_sum = 0.0f;
                    for (size_t c = 0; c < N; ++c) {
                        row_sum += accC[r][c];
                    }
                    // atomic add to device_sum
                    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space> a(device_sum[0]);
                    a.fetch_add(row_sum);
                });
            }).wait();
        }

        std::cout << "Device checksum = " << device_sum[0] << "\n";

        // 4) Copy C back and verify sample differences
        {
            auto acc = bufC.get_access<sycl::access::mode::read>();
            size_t errors = 0;
            double max_err = 0.0;
            for (size_t r = 0; r < M; ++r) {
                for (size_t c = 0; c < N; ++c) {
                    float got = acc[r][c];
                    float expected = C_ref[idx(r, c, N)];
                    double err = std::abs(got - expected);
                    if (err > 1e-3f && ++errors <= 10) {
                        std::cout << "Mismatch at (" << r << "," << c << ") got=" << got
                                  << " expected=" << expected << " err=" << err << "\n";
                    }
                    if (err > max_err) max_err = err;
                }
            }
            std::cout << "Verification: max error = " << max_err
                      << ", total error positions > 1e-3: " << errors << "\n";
        }

        // Cleanup
        sycl::free(A, q);
        sycl::free(B, q);
        sycl::free(Bt, q);
        sycl::free(device_sum, q);

        std::cout << "Done.\n";
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << " (" << e.code() << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "std exception: " << e.what() << "\n";
        return 2;
    }
    return 0;
}
