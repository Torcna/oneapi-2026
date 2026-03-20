#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue queue(device);
    
    float step = (end - start) / count;
    float result = 0.0f;
    
    {
        sycl::buffer<float> result_buf(&result, 1);
        
        queue.submit([&](sycl::handler& cgh) {
            auto result_acc = result_buf.get_access<sycl::access::mode::write>(cgh);
            
            cgh.parallel_for(sycl::range<2>(count, count), [=](sycl::id<2> idx) {
                int i = idx[0];
                int j = idx[1];
                
                float x = start + (i + 0.5f) * step;
                float y = start + (j + 0.5f) * step;
                
                float value = sycl::sin(x) * sycl::cos(y);
                
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device> atomic_result(result_acc[0]);
                atomic_result.fetch_add(value * step * step);
            });
        });
        
        queue.wait();
    }
    
    return result;
}