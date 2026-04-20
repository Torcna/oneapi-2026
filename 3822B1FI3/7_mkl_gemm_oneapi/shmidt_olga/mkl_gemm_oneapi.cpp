#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device)
{
    std::vector<float> c(size * size, 0.0f);

    sycl::queue queue(device);

    float* d_a = sycl::malloc_device<float>(size * size, queue);
    float* d_b = sycl::malloc_device<float>(size * size, queue);
    float* d_c = sycl::malloc_device<float>(size * size, queue);

    queue.memcpy(d_a, a.data(), size * size * sizeof(float));
    queue.memcpy(d_b, b.data(), size * size * sizeof(float));
    queue.memcpy(d_c, c.data(), size * size * sizeof(float));
    queue.wait();

    float alpha = 1.0f;
    float beta = 0.0f;

    oneapi::mkl::blas::column_major::gemm(
        queue,
        oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans,
        size, size, size,
        alpha,
        d_a, size,
        d_b, size,
        beta,
        d_c, size);

    queue.wait();

    queue.memcpy(c.data(), d_c, size * size * sizeof(float));
    queue.wait();

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_c, queue);

    return c;
}