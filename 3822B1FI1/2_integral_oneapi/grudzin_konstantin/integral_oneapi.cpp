#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float local_res = .0f;
  const float delta = (end - start) / count;

  sycl::queue q(device);

  {
    sycl::buffer<float> sum_buf(&local_res, 1);

    q.submit([&](sycl::handler &cgh) {
       auto reduction = sycl::reduction(sum_buf, cgh, sycl::plus<>());

       cgh.parallel_for(sycl::range<2>(count, count), reduction,
                        [=](sycl::id<2> id, auto &sum) {
                          float x = start + delta * (id.get(0) + 0.5f);
                          float y = start + delta * (id.get(1) + 0.5f);
                          sum += sycl::sin(x) * sycl::cos(y);
                        });
     }).wait();
  }

  return local_res * delta * delta;
}