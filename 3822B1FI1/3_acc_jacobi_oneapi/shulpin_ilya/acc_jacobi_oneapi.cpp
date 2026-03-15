#include "acc_jacobi_oneapi.h"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device)
{
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const size_t n = b.size();
    if (n == 0 || a.size() != n * n) {
        return {};
    }

    try {
        sycl::queue q{device};

        sycl::buffer<float, 1> A_buf{a.data(), sycl::range<1>{a.size()}};
        sycl::buffer<float, 1> b_buf{b.data(), sycl::range<1>{n}};

        std::vector<float> x_host(n, 0.0f);

        sycl::buffer<float, 1> x1_buf{x_host.data(), sycl::range<1>{n}};
        sycl::buffer<float, 1> x2_buf{sycl::range<1>{n}};

        sycl::buffer<float, 1>* x_current = &x1_buf;
        sycl::buffer<float, 1>* x_next = &x2_buf;

        bool converged = false;

        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            float max_diff = 0.0f;

            q.submit([&](sycl::handler& h)
            {
                auto A_acc   = A_buf.get_access<sycl::access::mode::read>(h);
                auto b_acc   = b_buf.get_access<sycl::access::mode::read>(h);
                auto x_cur_acc = x_current->get_access<sycl::access::mode::read>(h);
                auto x_new_acc = x_next->get_access<sycl::access::mode::write>(h);

                auto max_red = sycl::reduction(max_diff, sycl::maximum<float>());

                h.parallel_for(sycl::range<1>{n},
                    max_red,
                    [=](sycl::id<1> idx, auto& local_max)
                    {
                        const size_t i = idx[0];
                        float sum = 0.0f;

                        for (size_t j = 0; j < n; ++j)
                        {
                            if (j != i) {
                                sum += A_acc[i * n + j] * x_cur_acc[j];
                            }
                        }

                        float diag = A_acc[i * n + i];
                        if (std::abs(diag) < 1e-12f) {
                            x_new_acc[i] = x_cur_acc[i];
                        } else {
                            x_new_acc[i] = (b_acc[i] - sum) / diag;
                        }

                        float diff = sycl::fabs(x_new_acc[i] - x_cur_acc[i]);
                        local_max.combine(diff);
                    });
            }).wait();

            if (max_diff < accuracy) {
                converged = true;
                break;
            }

            std::swap(x_current, x_next);
        }

        sycl::host_accessor<float, 1, sycl::access::mode::read> result_acc(*x_current);
        std::vector<float> solution(n);
        for (size_t i = 0; i < n; ++i) {
            solution[i] = result_acc[i];
        }

        return solution;

    } catch (sycl::exception const& e) {
        return {};
    }
}