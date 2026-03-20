#include "block_gemm_oneapi.h"
#include <vector>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        size_t size, sycl::device device) {
    
    sycl::queue queue(device);
    
    size_t block_size = 32;
    size_t num_blocks = size / block_size;
    
    std::vector<float> c(size * size, 0.0f);
    
    float* a_dev = sycl::malloc_device<float>(a.size(), queue);
    float* b_dev = sycl::malloc_device<float>(b.size(), queue);
    float* c_dev = sycl::malloc_device<float>(c.size(), queue);
    
    queue.memcpy(a_dev, a.data(), a.size() * sizeof(float)).wait();
    queue.memcpy(b_dev, b.data(), b.size() * sizeof(float)).wait();
    queue.memset(c_dev, 0, c.size() * sizeof(float)).wait();
    
    for (size_t I = 0; I < num_blocks; ++I) {
        for (size_t J = 0; J < num_blocks; ++J) {
            for (size_t K = 0; K < num_blocks; ++K) {
                queue.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<2>(block_size, block_size), 
                                     [=](sycl::id<2> idx) {
                        size_t i = idx[0];
                        size_t j = idx[1];
                        
                        size_t global_i = I * block_size + i;
                        size_t global_j = J * block_size + j;
                        
                        float sum = 0.0f;
                        
                        for (size_t k = 0; k < block_size; ++k) {
                            size_t global_k = K * block_size + k;
                            sum += a_dev[global_i * size + global_k] * 
                                   b_dev[global_k * size + global_j];
                        }
                        
                        sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                          sycl::memory_scope::device> atomic_c(c_dev[global_i * size + global_j]);
                        atomic_c.fetch_add(sum);
                    });
                });
            }
        }
    }
    
    queue.wait();
    queue.memcpy(c.data(), c_dev, c.size() * sizeof(float)).wait();
    
    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(c_dev, queue);
    
    return c;
}