#include "dev_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    sycl::queue queue(device);
    
    int n = b.size();
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);
    
    float* a_dev = sycl::malloc_device<float>(a.size(), queue);
    float* b_dev = sycl::malloc_device<float>(b.size(), queue);
    float* x_dev = sycl::malloc_device<float>(n, queue);
    float* x_new_dev = sycl::malloc_device<float>(n, queue);
    
    queue.memcpy(a_dev, a.data(), a.size() * sizeof(float)).wait();
    queue.memcpy(b_dev, b.data(), b.size() * sizeof(float)).wait();
    queue.memset(x_dev, 0, n * sizeof(float)).wait();
    queue.memset(x_new_dev, 0, n * sizeof(float)).wait();
    
    bool converged = false;
    
    for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;
                float a_ii = a_dev[i * n + i];
                
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_dev[i * n + j] * x_dev[j];
                    }
                }
                
                x_new_dev[i] = (b_dev[i] - sum) / a_ii;
            });
        });
        
        queue.wait();
        
        float diff_norm = 0.0f;
        float* diff_dev = sycl::malloc_device<float>(1, queue);
        queue.memset(diff_dev, 0, sizeof(float)).wait();
        
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float diff = sycl::fabs(x_new_dev[i] - x_dev[i]);
                
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device> atomic_diff(diff_dev[0]);
                if (diff > atomic_diff.load()) {
                    atomic_diff.store(diff);
                }
            });
        });
        
        queue.wait();
        queue.memcpy(&diff_norm, diff_dev, sizeof(float)).wait();
        sycl::free(diff_dev, queue);
        
        std::swap(x, x_new);
        
        float* temp = x_dev;
        x_dev = x_new_dev;
        x_new_dev = temp;
        
        if (diff_norm < accuracy) {
            converged = true;
        }
    }
    
    queue.memcpy(x.data(), x_dev, n * sizeof(float)).wait();
    
    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(x_dev, queue);
    sycl::free(x_new_dev, queue);
    
    return x;
}