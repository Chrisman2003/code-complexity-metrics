#include <CL/sycl.hpp>
#include <iostream>

namespace sycl = cl::sycl;

struct Vec3 {
    float x, y, z;

    Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

int main() {
    sycl::queue q{sycl::gpu_selector{}};

    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    const int N = 8;
    Vec3 input[N];

    for (int i = 0; i < N; i++) {
        input[i] = { static_cast<float>(i), 1.0f, -1.0f };
    }

    Vec3 result{0.0f, 0.0f, 0.0f};

    {
        sycl::buffer<Vec3, 1> bufIn(input, sycl::range<1>(N));
        sycl::buffer<Vec3, 1> bufOut(&result, sycl::range<1>(1));

        q.submit([&](sycl::handler& h) {
            auto in = bufIn.get_access<sycl::access::mode::read>(h);
            auto out = bufOut.get_access<sycl::access::mode::write>(h);

            h.parallel_for<class SimpleKernel>(
                sycl::range<1>(N),
                [=](sycl::item<1> item) {
                    size_t i = item.get_linear_id();
                    Vec3 v = in[i];

                    // simple accumulation
                    out[0].x += v.x;
                    out[0].y += v.y * 2.0f;
                    out[0].z += v.z * -1.0f;
                }
            );
        });
    }

    std::cout << "Result = ("
              << result.x << ", "
              << result.y << ", "
              << result.z << ")\n";

    return 0;
}
