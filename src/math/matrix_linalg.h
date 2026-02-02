#pragma once
#include "matrix.h"
#include "utils/asserts.h"
#include <cassert>
#include <cstddef>
#include <vector>

namespace Math {

namespace Linalg {

template <typename T> Matrix<T> matmul(const Matrix<T> &a, const Matrix<T> &b) {
  assert_eq(a.shape().size(), (size_t)2,
            "Matrix::Linalg::Matmul::ValueError::Only support Matrix n x m");
  assert_eq(b.shape().size(), (size_t)2,
            "Matrix::Linalg::Matmul::ValueError::Only support Matrix n x m");

  assert_eq(a.shape()[1], b.shape()[0],
            "Matrix::Linalg::Matmul::ValueError::Dimesion mistmatch (cols A != "
            "rows B)");

  int rowsA{a.shape()[0]};
  int colsA{a.shape()[1]};
  int rowsB{b.shape()[0]};
  int colsB{b.shape()[1]};

  const T *pA = a.data_ptr();
  const T *pB = b.data_ptr();
  std::vector<T> result((size_t)(rowsA * colsB), 0);
  T *pResult = result.data();

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++) {

      int idResult = i * colsB + j;
      int idBaseA = i * colsA;

      T sum = 0;

      for (int k = 0; k < colsA; k++) {
        sum += pA[idBaseA + k] * pB[k * colsB + j];
      }

      pResult[idResult] = sum;
    }
  }

  return {result, {rowsA, colsB}};
}

template <typename T> Matrix<T> transpose(const Matrix<T> &matrix) {

  std::vector<T> out(matrix.size());

  T *pOut = out.data();
  const T *pMatrix = matrix.data_ptr();
  int idOut{0};
  int rows = matrix.shape()[0];
  int cols = matrix.shape()[1];

#pragma omp parallel for collapse(2)
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j < rows; j++) {

      int idOut = i * rows + j;
      int idIn = j * cols + i;

      pOut[idOut] = pMatrix[idIn];
    }
  }

  return {out, {cols, rows}};
}

template <typename T> Matrix<T> ones(std::vector<int> shape) {

  size_t sizeInt{1};
  for (const auto &element : shape) {
    assert_lineq(element, (int)1, "Matrix::Const::Dimesions must be positive");
    sizeInt *= (size_t)element;
  }

  std::vector<T> out(sizeInt, 1);

  return {out, shape};
}

template <typename T> Matrix<T> zeros(std::vector<int> shape) {

  size_t sizeInt{1};
  for (const auto &element : shape) {
    assert_lineq(element, (int)1, "Matrix::Const::Dimesions must be positive");
    sizeInt *= (size_t)element;
  }

  std::vector<T> out(sizeInt, 0);

  return {out, shape};
}

template <typename T> Matrix<T> sum(const Matrix<T> &matrix, size_t axis) {
  assert_gineq(axis, (size_t)1,
               "Matrix::LineAlg::Sum::ValueError:::Index out of bounds");

  int nrows = matrix.shape()[0];
  int ncols = matrix.shape()[1];

  // Out size
  size_t resultSize = (axis == 0) ? ncols : nrows;

  std::vector<T> result(resultSize, 0);

  T *pResult = result.data();
  const T *pMatrix = matrix.data_ptr();

  // --- CASO AXIS 0: Sum on Rows
  if (axis == 0) {
#pragma omp parallel for
    for (int j = 0; j < ncols; j++) {
      T acc = 0;
      for (int i = 0; i < nrows; i++) {
        acc += pMatrix[i * ncols + j];
      }
      pResult[j] = acc;
    }
  }

  // --- CASO AXIS 1: Sum on Columns
  else {
#pragma omp parallel for
    for (int i = 0; i < nrows; i++) {
      T acc = 0;
      int baseIdx = i * ncols;
      for (int j = 0; j < ncols; j++) {
        acc += pMatrix[baseIdx + j];
      }
      pResult[i] = acc;
    }
  }

  // Shape Out
  std::vector<int> shapeResult =
      (axis == 0) ? std::vector<int>{1, ncols} : std::vector<int>{nrows, 1};

  return {result, shapeResult};
}

// Sum all the elements from a Matrix m and return a Matrix 1x1 with the sum
template <typename T> Matrix<T> sum(const Matrix<T> &m) {
  T sum{(T)0};
  for (const auto &element : m.data()) {
    sum += element;
  }

  return Matrix<T>(std::vector<T>({sum}), std::vector<int>({1, 1}));
}

} // namespace Linalg
} // namespace Math
