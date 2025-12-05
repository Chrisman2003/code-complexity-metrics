#include <iostream>
#include <vector>
#include <boost/compute.hpp>

int main()
{
    // 1. Select default compute device
    boost::compute::device device = boost::compute::system::default_device();
    std::cout << "Using device: " << device.name() << std::endl;

    // 2. Create context + queue for this device
    boost::compute::context context(device);
    boost::compute::command_queue queue(context, device);

    // 3. Host input data
    const int N = 16;
    std::vector<int> host_input(N);
    for(int i = 0; i < N; i++)
        host_input[i] = i;

    // 4. Device buffers
    boost::compute::vector<int> d_input(N, context);
    boost::compute::vector<int> d_output(N, context);

    // 5. Copy to device (host → GPU)
    boost::compute::copy(
        host_input.begin(),
        host_input.end(),
        d_input.begin(),
        queue
    );

    // 6. Parallel transform using Boost.Compute constructs
    //    d_output[i] = d_input[i] * d_input[i]
    boost::compute::transform(
        d_input.begin(),
        d_input.end(),
        d_output.begin(),
        boost::compute::_1 * boost::compute::_1,  // <- computed on GPU
        queue
    );

    // 7. Copy results back (GPU → host)
    std::vector<int> result(N);
    boost::compute::copy(
        d_output.begin(),
        d_output.end(),
        result.begin(),
        queue
    );

    // 8. Print results
    std::cout << "Squares: ";
    for(int v : result)
        std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
