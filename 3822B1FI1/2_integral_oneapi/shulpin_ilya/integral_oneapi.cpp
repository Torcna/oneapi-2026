#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) return 0.0f;

    const float dx = (end - start) / static_cast<float>(count);
    const float dy = dx;

    float result = 0.0f;

    try {
        sycl::queue q(device);

        sycl::buffer<float, 0> sum_buf{&result, sycl::range<0>{}};

        q.submit([&](sycl::handler& h) {
            auto sum_acc = sum_buf.get_access<sycl::access::mode::read_write>(h);

            h.parallel_reduce(
                sycl::range<2>(count, count),
                sycl::reduction(sum_acc, sycl::plus<float>()),
                [=](sycl::nd_item<2> item, auto& sum) {
                    const int i = item.get_global_id(1);
                    const int j = item.get_global_id(0);

                    const float x = start + (i + 0.5f) * dx;
                    const float y = start + (j + 0.5f) * dy;

                    sum += sycl::sin(x) * sycl::cos(y) * dx * dy;
                });
        }).wait();
    }
    catch (sycl::exception const&) {
        return 0.0f;
    }

    return result;
}