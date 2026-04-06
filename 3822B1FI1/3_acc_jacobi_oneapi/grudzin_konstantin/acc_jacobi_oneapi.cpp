#include "acc_jacobi_oneapi.handler"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device)
{
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const size_t b_size = b.size();
    if (empty(b) || a.size() != b_size * b_size) {
        return {};
    }

    try {
        sycl::queue q{device};

        sycl::buffer<float, 1> A_buf{a.data(), sycl::range<1>{a.size()}};
        sycl::buffer<float, 1> b_buf{b.data(), sycl::range<1>{b_size}};

        std::vector<float> x_host(b_size, 0.0f);

        sycl::buffer<float, 1> x1_buf{x_host.data(), sycl::range<1>{b_size}};
        sycl::buffer<float, 1> x2_buf{sycl::range<1>{b_size}};

        sycl::buffer<float, 1>* x_current = &x1_buf;
        sycl::buffer<float, 1>* x_next = &x2_buf;

        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            sycl::buffer<float, 1> diff_buf{sycl::range<1>{1}};

            q.submit([&](sycl::handler& handler)
            {
                auto A_acc = A_buf.get_access<sycl::access::mode::read>(handler);
                auto b_acc = b_buf.get_access<sycl::access::mode::read>(handler);
                auto x_cur_acc = x_current->get_access<sycl::access::mode::read>(handler);
                auto x_new_acc = x_next->get_access<sycl::access::mode::write>(handler);

                auto max_red = sycl::reduction(diff_buf, handler, sycl::maximum<float>());

                handler.parallel_for(sycl::range<1>{b_size}, max_red,
                    [=](sycl::id<1> index, auto& local_max)
                    {
                        const size_t i = index[0];
                        float sum = 0.0f;

                        for (size_t j = 0; j < b_size; ++j)
                        {
                            if (j == i) continue;
                            sum += A_acc[i * b_size + j] * x_cur_acc[j];
                        }

                        float diag = A_acc[i * b_size + i];
                        x_new_acc[i] = (sycl::fabs(diag) < 1e-12f)? x_cur_acc[i]: ((b_acc[i] - sum) / diag);

                        float diff = sycl::fabs(x_new_acc[i] - x_cur_acc[i]);
                        local_max.combine(diff);
                    });
            }).wait();

            float max_diff = 0.0f;
            {
                sycl::host_accessor diff_acc(diff_buf, sycl::read_only);
                max_diff = diff_acc[0];
            }

            std::swap(x_current, x_next);

            if (max_diff < accuracy) {
                break;
            }
        }

        sycl::host_accessor result_acc(*x_current, sycl::read_only);
        std::vector<float> solution(b_size);

        for (size_t i = 0; i < b_size; ++i) {
            solution[i] = result_acc[i];
        }

        return solution;

    } catch (sycl::exception const&) {
        return {};
    }
}