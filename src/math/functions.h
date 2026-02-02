#include "matrix.h"
#include <cmath>
#include <vector>

namespace Math {
namespace Func {

// Apply a function to a Matrix Element-Wise
template <typename T, typename F> Matrix<T> apply(const Matrix<T> &m, F func) {
  size_t size = m.size();
  std::vector<T> result(size);

  const T *pIn = m.data_ptr();
  T *pOut = result.data();

#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    pOut[i] = func(pIn[i]);
  }

  return {result, m.shape()};
}

// Exp
template <typename T> Matrix<T> exp(const Matrix<T> &m) {
  return apply<T>(m, [](T x) { return std::exp(x); });
}

// log
template <typename T> Matrix<T> log(const Matrix<T> &m) {
  return apply<T>(m, [](T x) { return std::log(x); });
}

// Sigmoid
template <typename T> Matrix<T> sigmoid(const Matrix<T> &m) {
  return (T)1.0 / ((T)1.0 + exp((T)-1.0 * m));
}

// Tanh
template <typename T> Matrix<T> tanh(const Matrix<T> &m) {
  return apply<T>(m, [](T x) { return std::tanh(x); });
}

// Pow
template <typename T> Matrix<T> pow(const Matrix<T> &m, T power) {
  return apply<T>(m, [power](T x) { return std::pow(x, power); });
}

// ReLU
template <typename T> Matrix<T> relu(const Matrix<T> &m) {
  return apply<T>(m, [](T x) { return x > (T)0 ? x : (T)0; });
}

} // namespace Func
} // namespace Math
