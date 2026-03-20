#include "shared_jacobi_oneapi.h"
#include <cmath>
#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    
    sycl::queue queue(device);
    
    int n = b.size();
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);
    
    float* a_shared = sycl::malloc_shared<float>(a.size(), queue);
    float* b_shared = sycl::malloc_shared<float>(b.size(), queue);
    float* x_shared = sycl::malloc_shared<float>(n, queue);
    float* x_new_shared = sycl::malloc_shared<float>(n, queue);
    
    queue.memcpy(a_shared, a.data(), a.size() * sizeof(float)).wait();
    queue.memcpy(b_shared, b.data(), b.size() * sizeof(float)).wait();
    
    for (int i = 0; i < n; ++i) {
        x_shared[i] = 0.0f;
        x_new_shared[i] = 0.0f;
    }
    
    bool converged = false;
    
    for (int iter = 0; iter < ITERATIONS && !converged; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float sum = 0.0f;
                float a_ii = a_shared[i * n + i];
                
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_shared[i * n + j] * x_shared[j];
                    }
                }
                
                x_new_shared[i] = (b_shared[i] - sum) / a_ii;
            });
        });
        
        queue.wait();
        
        float diff_norm = 0.0f;
        
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = idx[0];
                float diff = sycl::fabs(x_new_shared[i] - x_shared[i]);
                
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::system> atomic_diff(diff_norm);
                if (diff > atomic_diff.load()) {
                    atomic_diff.store(diff);
                }
            });
        });
        
        queue.wait();
        
        std::swap(x, x_new);
        std::swap(x_shared, x_new_shared);
        
        if (diff_norm < accuracy) {
            converged = true;
        }
    }
    
    for (int i = 0; i < n; ++i) {
        x[i] = x_shared[i];
    }
    
    sycl::free(a_shared, queue);
    sycl::free(b_shared, queue);
    sycl::free(x_shared, queue);
    sycl::free(x_new_shared, queue);
    
    return x;
}