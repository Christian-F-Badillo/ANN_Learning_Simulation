#include "matrix.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T> Matrix<T> matmul(const Matrix<T> &a, const Matrix<T> &b) {
  if (a.shape().size() != 2 || b.shape().size() != 2) {
    throw std::invalid_argument("Matmul: Only support Matrix n x m");
  }

  if (a.shape()[1] != b.shape()[0]) {
    throw std::invalid_argument(
        "Matmul: Dimesion mistmatch (cols A != rows B)");
  }

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
    if (element <= 0)
      throw std::invalid_argument(
          "Matrix: shape dimensions cannot be negative.");
    sizeInt *= (size_t)element;
  }

  std::vector<T> out(sizeInt, 1);

  return {out, shape};
}

template <typename T> Matrix<T> zeros(std::vector<int> shape) {

  size_t sizeInt{1};
  for (const auto &element : shape) {
    if (element <= 0)
      throw std::invalid_argument(
          "Matrix: shape dimensions cannot be negative.");
    sizeInt *= (size_t)element;
  }

  std::vector<T> out(sizeInt, 0);

  return {out, shape};
}
