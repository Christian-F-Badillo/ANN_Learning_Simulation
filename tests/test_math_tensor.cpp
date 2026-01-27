#include "../src/math/tensor.h"
#include "test_utils.h"
#include <cstddef>

int main() {

  TEST_CASE("Tensor Creation");
  Matrix<double> tensor1({1, 2, 3, 4}, {2, 2});
  Matrix<double> tensor2({1, 1, 1, 1}, {2, 2});

  const double *pTensor1 = tensor1.data_ptr();
  const double *pTensor2 = tensor2.data_ptr();

  Matrix<double> sum = tensor1 + tensor2;
  Matrix<double> rest = tensor1 - tensor2;
  Matrix<double> scalar_product = tensor1 * (double)2.0;

  const double *pSum = sum.data_ptr();
  const double *pRest = rest.data_ptr();
  const double *pScalarProd = scalar_product.data_ptr();

  TEST_CASE("Tensor Sum");
#pragma omp simd
  for (size_t i = 0; i < sum.size(); i++) {
    ASSERT_ALMOST_EQ(pSum[i], pTensor1[i] + pTensor2[i]);
  }

  TEST_CASE("Tensor Difference");
#pragma omp simd
  for (size_t i = 0; i < rest.size(); i++) {
    ASSERT_ALMOST_EQ(pRest[i], pTensor1[i] - pTensor2[i]);
  }

  TEST_CASE("Tensor Scalar Product");
  double scalar{2.0};
#pragma omp simd
  for (size_t i = 0; i < scalar_product.size(); i++) {
    ASSERT_ALMOST_EQ(pScalarProd[i], pTensor1[i] * scalar);
  }

  return run_test_summary();
}
