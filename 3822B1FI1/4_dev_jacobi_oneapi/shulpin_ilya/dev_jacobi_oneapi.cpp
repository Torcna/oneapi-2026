#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a,
        const std::vector<float> b,
        float accuracy,
        sycl::device device)
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

        float* A_dev   = sycl::malloc_device<float>(a.size(), q);
        float* b_dev   = sycl::malloc_device<float>(n, q);
        float* x_curr  = sycl::malloc_device<float>(n, q);
        float* x_next  = sycl::malloc_device<float>(n, q);

        float* max_diff_dev = sycl::malloc_device<float>(1, q);

        q.memcpy(A_dev, a.data(), a.size() * sizeof(float)).wait();
        q.memcpy(b_dev, b.data(), n * sizeof(float)).wait();
        q.memcpy(x_curr, std::vector<float>(n, 0.0f).data(), n * sizeof(float)).wait();
        q.memcpy(x_next, x_curr, n * sizeof(float)).wait();

        bool converged = false;

        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            q.memset(max_diff_dev, 0, sizeof(float)).wait();

            sycl::buffer<float, 1> diff_buf{&max_diff_host, sycl::range<1>{1}};

            q.submit([&](sycl::handler& h)
            {
                auto max_red = sycl::reduction(diff_buf, h, sycl::maximum<float>());

                h.parallel_for(
                    sycl::range<1>{n},
                    max_red,
                    [=](sycl::id<1> id, auto& local_max)
                    {
                        const size_t i = id[0];
                        float sigma = 0.0f;

                        for (size_t j = 0; j < n; ++j)
                        {
                            if (j != i)
                            {
                                sigma += A_dev[i * n + j] * x_curr[j];
                            }
                        }

                        float diag = A_dev[i * n + i];
                        float new_val = (sycl::fabs(diag) < 1e-12f)
                                        ? x_curr[i]
                                        : (b_dev[i] - sigma) / diag;

                        x_next[i] = new_val;

                        float diff = sycl::fabs(new_val - x_curr[i]);
                        local_max.combine(diff);
                    });
            }).wait();

            float max_diff_host = 0.0f;
            q.memcpy(&max_diff_host, max_diff_dev, sizeof(float)).wait();

            if (max_diff_host < accuracy)
            {
                converged = true;
                break;
            }

            std::swap(x_curr, x_next);
        }

        std::vector<float> solution(n);
        q.memcpy(solution.data(), x_curr, n * sizeof(float)).wait();

        sycl::free(A_dev,   q);
        sycl::free(b_dev,   q);
        sycl::free(x_curr,  q);
        sycl::free(x_next,  q);
        sycl::free(max_diff_dev, q);

        return solution;

    }
    catch (sycl::exception const&)
    {
        return {};
    }
}