#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    const float delta = (end - start) / static_cast<float>(count);
    float total_integral = 0.0f;

    using DeviceSpace = Kokkos::SYCL;

    Kokkos::MDRangePolicy<DeviceSpace, Kokkos::Rank<2>> compute_policy(
        {0, 0},
        {count, count}
    );

    Kokkos::parallel_reduce("DoubleIntegralReduction", compute_policy,
        KOKKOS_LAMBDA(const int i, const int j, float& lsum) {
            const float x = start + (static_cast<float>(i) + 0.5f) * delta;
            const float y = start + (static_cast<float>(j) + 0.5f) * delta;

            lsum += Kokkos::sin(x) * Kokkos::cos(y);
        }, 
        total_integral
    );

    Kokkos::fence();

    return total_integral * (delta * delta);
}