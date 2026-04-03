#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    const float step_size = (end - start) / static_cast<float>(count);
    float result = 0.0f;
    
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::SYCL>(0, count * count),
        KOKKOS_LAMBDA(int id, float& accumulator) {
            int idx = id / count;
            int idi = id % count;
            float midpoint_x = start + step_size * (static_cast<float>(idx) + 0.5f);
            float midpoint_y = start + step_size * (static_cast<float>(idi) + 0.5f);
            accumulator += std::sin(midpoint_x) * std::cos(midpoint_y)*step_size*step_size;
        },
        total_sum
    );
    
    Kokkos::deep_copy(result, total_sum);

    return result;
}