#include "dev_jacobi_oneapi.h"

#include <algorithm>
#include <cmath>

std::vector<float> JacobiDevONEAPI(const std::vector<float> &a, const std::vector<float> &b, float accuracy,
                                   sycl::device device)
{
    const size_t n = b.size();

    sycl::queue queue(device);

    float *matrix = sycl::malloc_device<float>(n * n, queue);
    float *rhs = sycl::malloc_device<float>(n, queue);
    float *current = sycl::malloc_device<float>(n, queue);
    float *next = sycl::malloc_device<float>(n, queue);

    queue.memcpy(matrix, a.data(), sizeof(float) * n * n).wait();
    queue.memcpy(rhs, b.data(), sizeof(float) * n).wait();
    queue.memset(current, 0, sizeof(float) * n).wait();
    queue.memset(next, 0, sizeof(float) * n).wait();

    std::vector<float> prev_host(n, 0.0f);
    std::vector<float> next_host(n, 0.0f);

    bool converged = false;

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        queue
            .parallel_for(sycl::range<1>(n),
                          [=](sycl::id<1> idx) {
                              const size_t row = idx[0];

                              float sum = 0.0f;
                              for (size_t col = 0; col < n; ++col)
                              {
                                  if (col != row)
                                  {
                                      sum += matrix[row * n + col] * current[col];
                                  }
                              }

                              next[row] = (rhs[row] - sum) / matrix[row * n + row];
                          })
            .wait();

        queue.memcpy(next_host.data(), next, sizeof(float) * n).wait();

        float max_error = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            max_error = std::max(max_error, std::fabs(next_host[i] - prev_host[i]));
        }

        if (max_error < accuracy)
        {
            converged = true;
            break;
        }

        prev_host = next_host;

        float *tmp = current;
        current = next;
        next = tmp;
    }

    std::vector<float> result(n);
    if (converged)
    {
        result = next_host;
    }
    else
    {
        queue.memcpy(result.data(), current, sizeof(float) * n).wait();
    }

    sycl::free(matrix, queue);
    sycl::free(rhs, queue);
    sycl::free(current, queue);
    sycl::free(next, queue);

    return result;
}
