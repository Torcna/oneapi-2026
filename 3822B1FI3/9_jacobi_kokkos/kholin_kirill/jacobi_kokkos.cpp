#include "jacobi_kokkos.h"

#include <cmath>
#include <algorithm>

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy) {
    using ExecSpace = Kokkos::SYCL;
    using MemSpace = Kokkos::SYCLDeviceUSMSpace;

    const int n = static_cast<int>(b.size());
    if (n <= 0) {
        return {};
    }

    Kokkos::View<float**, Kokkos::LayoutLeft, MemSpace> matrix("matrix", n, n);
    Kokkos::View<float*, MemSpace> rhs("rhs", n);
    Kokkos::View<float*, MemSpace> inv_diag("inv_diag", n);
    Kokkos::View<float*, MemSpace> curr("curr", n);
    Kokkos::View<float*, MemSpace> next("next", n);

    auto host_matrix = Kokkos::create_mirror_view(matrix);
    auto host_rhs = Kokkos::create_mirror_view(rhs);

    for (int i = 0; i < n; ++i) {
        host_rhs(i) = b[i];
        for (int j = 0; j < n; ++j) {
            host_matrix(i, j) = a[i * n + j];
        }
    }

    Kokkos::deep_copy(matrix, host_matrix);
    Kokkos::deep_copy(rhs, host_rhs);

    Kokkos::parallel_for(
        "JacobiInit",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int i) {
            inv_diag(i) = 1.0f / matrix(i, i);
            curr(i) = 0.0f;
            next(i) = 0.0f;
        }
    );

    const int check_period = 4;
    bool done = false;
    int iter = 0;

    for (; iter < ITERATIONS && !done; ++iter) {
        Kokkos::parallel_for(
            "JacobiSweep",
            Kokkos::RangePolicy<ExecSpace>(0, n),
            KOKKOS_LAMBDA(const int row) {
                float sum = 0.0f;

                for (int col = 0; col < n; ++col) {
                    if (col != row) {
                        sum += matrix(row, col) * curr(col);
                    }
                }

                next(row) = (rhs(row) - sum) * inv_diag(row);
            }
        );

        if ((iter + 1) % check_period == 0) {
            float max_delta = 0.0f;

            Kokkos::parallel_reduce(
                "JacobiResidual",
                Kokkos::RangePolicy<ExecSpace>(0, n),
                KOKKOS_LAMBDA(const int i, float& local_max) {
                    const float diff = Kokkos::fabs(next(i) - curr(i));
                    if (diff > local_max) {
                        local_max = diff;
                    }
                },
                Kokkos::Max<float>(max_delta)
            );

            if (max_delta < accuracy) {
                done = true;
            }
        }

        Kokkos::kokkos_swap(curr, next);
    }

    auto host_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), curr);

    std::vector<float> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = host_result(i);
    }

    return result;
}