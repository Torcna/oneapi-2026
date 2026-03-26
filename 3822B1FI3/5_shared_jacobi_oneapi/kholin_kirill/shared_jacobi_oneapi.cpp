#include "shared_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

std::vector<float> JacobiSharedONEAPI(const std::vector<float> &matrix_a,
                                      const std::vector<float> &vector_b,
                                      float accuracy, sycl::device device) {

  int matrix_size = vector_b.size();

  sycl::queue queue(device);

  float *shared_matrix =
      sycl::malloc_shared<float>(matrix_size * matrix_size, queue);
  float *shared_rhs = sycl::malloc_shared<float>(matrix_size, queue);
  float *shared_current = sycl::malloc_shared<float>(matrix_size, queue);
  float *shared_next = sycl::malloc_shared<float>(matrix_size, queue);

  for (int i = 0; i < matrix_size * matrix_size; ++i) {
    shared_matrix[i] = matrix_a[i];
  }

  for (int i = 0; i < matrix_size; ++i) {
    shared_rhs[i] = vector_b[i];
    shared_current[i] = 0.0f;
    shared_next[i] = 0.0f;
  }

  std::vector<float> diagonal(matrix_size);
  for (int i = 0; i < matrix_size; ++i) {
    diagonal[i] = matrix_a[i * matrix_size + i];
  }
  float *shared_diagonal = sycl::malloc_shared<float>(matrix_size, queue);
  for (int i = 0; i < matrix_size; ++i) {
    shared_diagonal[i] = diagonal[i];
  }

  for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
    queue
        .parallel_for(sycl::range<1>(matrix_size),
                      [=](sycl::id<1> idx) {
                        int row = idx[0];
                        float sum = 0.0f;

                        for (int col = 0; col < matrix_size; ++col) {
                          if (col != row) {
                            sum += shared_matrix[row * matrix_size + col] *
                                   shared_current[col];
                          }
                        }

                        shared_next[row] =
                            (shared_rhs[row] - sum) / shared_diagonal[row];
                      })
        .wait();

    float max_diff = 0.0f;
    for (int i = 0; i < matrix_size; ++i) {
      float diff = std::abs(shared_next[i] - shared_current[i]);
      max_diff = std::max(max_diff, diff);
      shared_current[i] = shared_next[i];
    }

    if (max_diff < accuracy) {
      break;
    }
  }

  std::vector<float> result(matrix_size);
  for (int i = 0; i < matrix_size; ++i) {
    result[i] = shared_current[i];
  }

  sycl::free(shared_matrix, queue);
  sycl::free(shared_rhs, queue);
  sycl::free(shared_current, queue);
  sycl::free(shared_next, queue);
  sycl::free(shared_diagonal, queue);

  return result;
}