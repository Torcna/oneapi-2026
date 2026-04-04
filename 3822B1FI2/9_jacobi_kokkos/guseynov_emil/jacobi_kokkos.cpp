#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(const std::vector<float>& mat_a,
                                const std::vector<float>& vec_b,
                                float eps) {
    const int n_size = static_cast<int>(vec_b.size());
    std::vector<float> solution(n_size);

    Kokkos::View<float*> v_a("matrix_a", n_size * n_size);
    Kokkos::View<float*> v_b("vector_b", n_size);
    Kokkos::View<float*> v_x("curr_x", n_size);
    Kokkos::View<float*> v_x_next("next_x", n_size);

    auto host_a = Kokkos::create_mirror_view(v_a);
    auto host_b = Kokkos::create_mirror_view(v_b);

    for (int i = 0; i < n_size; ++i) {
        host_b(i) = vec_b[i];
        for (int j = 0; j < n_size; ++j) {
            host_a(i * n_size + j) = mat_a[i * n_size + j];
        }
    }

    Kokkos::deep_copy(v_a, host_a);
    Kokkos::deep_copy(v_b, host_b);
    Kokkos::deep_copy(v_x, 0.0f);

    int step = 0;
    float current_err = 0.0f;

    do {
        Kokkos::parallel_for("JacobiStep", n_size, KOKKOS_LAMBDA(const int i) {
            float sum = 0.0f;
            for (int j = 0; j < n_size; ++j) {
                if (i != j) {
                    sum += v_a(i * n_size + j) * v_x(j);
                }
            }
            v_x_next(i) = (v_b(i) - sum) / v_a(i * n_size + i);
        });

        Kokkos::parallel_reduce("ErrorCalc", n_size, KOKKOS_LAMBDA(const int i, float& local_max) {
            float diff = Kokkos::fabs(v_x_next(i) - v_x(i));
            if (diff > local_max) local_max = diff;
        }, Kokkos::Max<float>(current_err));

        Kokkos::deep_copy(v_x, v_x_next);
        
        step++;
    } while (step < ITERATIONS && current_err >= eps);

    auto host_res = Kokkos::create_mirror_view(v_x);
    Kokkos::deep_copy(host_res, v_x);
    
    for (int i = 0; i < n_size; ++i) {
        solution[i] = host_res(i);
    }

    return solution;
}