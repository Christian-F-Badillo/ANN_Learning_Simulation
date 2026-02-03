#pragma once
#include "../utils/asserts.h"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace Math {

template <typename T> class Matrix;

// friend operators-declarations
template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix);

template <typename T>
Matrix<T> operator+(const Matrix<T> &left, const Matrix<T> &right);

template <typename T>
Matrix<T> operator+(const Matrix<T> &left, const std::vector<T> &vector);

template <typename T>
Matrix<T> operator+(const std::vector<T> &vector, const Matrix<T> &left);

template <typename T>
Matrix<T> operator+(const T &scalar, const Matrix<T> &matrix);

template <typename T>
Matrix<T> operator+(const Matrix<T> &matrix, const T &scalar);

template <typename T>
Matrix<T> operator-(const Matrix<T> &left, const Matrix<T> &right);

template <typename T>
Matrix<T> operator-(const T &scalar, const Matrix<T> &matrix);

template <typename T>
Matrix<T> operator-(const Matrix<T> &matrix, const T &scalar);

template <typename T>
Matrix<T> operator*(const T &scalar, const Matrix<T> &matrix);

template <typename T>
Matrix<T> operator*(const Matrix<T> &right, const T &scalar);

template <typename T>
Matrix<T> operator*(const Matrix<T> &left, const Matrix<T> &right);

template <typename T>
Matrix<T> operator/(const T &scalar, const Matrix<T> &matrix);

template <typename T>
Matrix<T> operator/(const Matrix<T> &right, const T &scalar);

template <typename T>
Matrix<T> operator/(const Matrix<T> &left, const Matrix<T> &right);

// Class Matrix implementation

template <typename T> class Matrix {

public:
  Matrix(std::vector<T> vectorIn, const std::vector<int> &shapeIn);
  Matrix(const std::vector<std::vector<T>> &matrix,
         const std::vector<int> &shapeIn);
  Matrix(const Matrix<T> &other);
  Matrix(Matrix<T> &&other) noexcept;
  friend std::ostream &operator<< <>(std::ostream &os, const Matrix<T> &matrix);
  friend Matrix<T> operator+ <>(const Matrix<T> &left, const Matrix<T> &right);
  friend Matrix<T> operator+
      <>(const Matrix<T> &left, const std::vector<T> &bias);
  friend Matrix<T> operator+
      <>(const std::vector<T> &bias, const Matrix<T> &left);
  friend Matrix<T> operator+ <>(const T &scalar, const Matrix<T> &matrix);
  friend Matrix<T> operator+ <>(const Matrix<T> &matrix, const T &scalar);
  friend Matrix<T> operator- <>(const Matrix<T> &left, const Matrix<T> &right);
  friend Matrix<T> operator- <>(const T &scalar, const Matrix<T> &matrix);
  friend Matrix<T> operator- <>(const Matrix<T> &matrix, const T &scalar);
  friend Matrix<T> operator* <>(const T &scalar, const Matrix<T> &matrix);
  friend Matrix<T> operator* <>(const Matrix<T> &right, const T &scalar);
  friend Matrix<T> operator* <>(const Matrix<T> &left, const Matrix<T> &right);
  friend Matrix<T> operator/ <>(const T &scalar, const Matrix<T> &matrix);
  friend Matrix<T> operator/ <>(const Matrix<T> &right, const T &scalar);
  friend Matrix<T> operator/ <>(const Matrix<T> &left, const Matrix<T> &right);
  Matrix<T> &operator=(Matrix<T> &&other) noexcept;
  Matrix<T> &operator=(const Matrix<T> &other);

  const size_t &size() const;
  const std::vector<int> &shape() const;
  const std::vector<T> &data() const;
  const T *data_ptr() const;
  Matrix<T> &reshape(const std::vector<int> &new_shape);
  Matrix<T> &view(std::vector<int> new_shape);
  T at(size_t row, size_t col);
  Matrix<T> atRow(size_t row) const;
  Matrix<T> atCol(size_t col) const;

private:
  std::vector<T> _data;
  std::vector<int> _shape;
  size_t _size{};

  size_t _getSize(const std::vector<int> &shapeIn);
  std::vector<T> _squeezeMatrix(const std::vector<std::vector<T>> &matrix);
  void print_recursive(std::ostream &os, size_t dim_index, size_t &offset,
                       size_t indent_level) const;
};

// **************************************
// Constructors
// **************************************
template <typename T>
Matrix<T>::Matrix(std::vector<T> vectorIn, const std::vector<int> &shapeIn)
    : _data(std::move(vectorIn)), _shape(shapeIn), _size(_getSize(shapeIn)) {
  assert_eq(_shape.size(), (size_t)2, "Matrix::Const::Dimension mismatch");
  assert_eq(_size, _data.size(), "Matrix::Const::ValueError.");
}
template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &matrix,
                  const std::vector<int> &shapeIn)
    : _data(_squeezeMatrix(matrix)), _shape(shapeIn), _size(_getSize(shapeIn)) {
  assert_eq(_shape.size(), (size_t)2, "Matrix::Const::Dimension mismatch");
  assert_eq(_size, _data.size(), "Matrix::Const::ValueError.");
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other)
    : _data(other._data), _shape(other._shape), _size(other._size) {}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &&other) noexcept
    : _data(std::move(other._data)), _shape(std::move(other._shape)),
      _size(other._size) {
  other._size = 0;
}
// ***************************************************************
// Utils Methods
// ***************************************************************
template <typename T>
size_t Matrix<T>::_getSize(const std::vector<int> &shapeIn) {
  size_t sizeInt{1};
  for (const auto &element : shapeIn) {
    assert_lineq(element, 0, "Matrix: shape dimensions cannot be negative.");
    sizeInt *= (size_t)element;
  }
  return sizeInt;
}

template <typename T>
std::vector<T>
Matrix<T>::_squeezeMatrix(const std::vector<std::vector<T>> &matrix) {

  std::vector<T> output;

  size_t totalElements = 0;
  for (const auto &row : matrix)
    totalElements += row.size();
  output.reserve(totalElements);

  for (const auto &row : matrix)
    output.insert(output.end(), row.begin(), row.end());
  return output;
}

template <typename T>
void Matrix<T>::print_recursive(std::ostream &os, size_t dim_index,
                                size_t &offset, size_t indent_level) const {
  if (shape().empty())
    return;

  int current_dim_size = shape()[dim_index];
  bool is_last_dim = (dim_index == shape().size() - 1);

  std::string indent(indent_level * 2, ' ');

  os << "[";

  if (!is_last_dim) {
    os << "\n";
  }

  for (int i = 0; i < current_dim_size; ++i) {
    if (!is_last_dim) {
      os << indent << "  ";
      print_recursive(os, dim_index + 1, offset, indent_level + 1);
    } else {
      os << data()[offset];
      offset++;
    }

    if (i < current_dim_size - 1) {
      os << ", ";
      if (!is_last_dim)
        os << "\n";
    }
  }

  if (!is_last_dim) {
    os << "\n" << indent;
  }
  os << "]";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
  os << "Matrix(";
  if (matrix.data().empty()) {
    os << "[]";
  } else {
    size_t offset = 0;
    matrix.print_recursive(os, 0, offset, 0);
  }
  os << ", shape=(";
  for (size_t i = 0; i < matrix.shape().size(); ++i) {
    os << matrix.shape()[i] << (i < matrix.shape().size() - 1 ? "," : "");
  }
  os << "))";

  return os;
}

/********************************************************************************
 *
 * Getters
 *
 *******************************************************************************/

template <typename T> const size_t &Matrix<T>::size() const { return _size; }
template <typename T> const std::vector<int> &Matrix<T>::shape() const {
  return _shape;
}
template <typename T> const std::vector<T> &Matrix<T>::data() const {
  return _data;
}
template <typename T> const T *Matrix<T>::data_ptr() const {
  return _data.data();
}

/*****************************************************
 *
 * Math Methods
 *
 ****************************************************/

template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept {
  if (this != &other) {
    _data = std::move(other._data);
    _shape = std::move(other._shape);
    _size = other._size;
    other._size = 0;
  }
  return *this;
}

template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other) {
  if (this == &other) {
    return *this;
  }

  this->_data = other._data;
  this->_shape = other._shape;
  this->_size = other._size;

  return *this;
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &left, const Matrix<T> &right) {

  if (left.shape() == right.shape()) { // Shall do Matrix sum?

    size_t size{left.size()};
    std::vector<T> sum(size);

    const T *pLeft = left.data_ptr();
    const T *pRight = right.data_ptr();
    T *pSum = sum.data();

#pragma omp simd
    for (size_t i = 0; i < size; i++) {
      pSum[i] = pLeft[i] + pRight[i];
    }

    return {sum, left.shape()};
  }

  // If not we shall do Broadcasting Sum
  else if (right.shape()[0] == 1 && right.shape()[1] == left.shape()[1]) {

    return left + right.data();
  } else if (left.shape()[0] == 1 && left.shape()[1] == right.shape()[1]) {
    return right + left.data();
  }
  // Else throw an error
  else {
    throw std::invalid_argument("Dimension mismatch: Shapes are incompatible "
                                "for Element-wise or Broadcast sum.");
  }
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &matrix, const std::vector<T> &bias) {

  assert_eq(bias.size(), (size_t)matrix.shape()[1], "BroadcastAdd::ValueError");

  std::vector<T> output = matrix.data();

  T *pOut = output.data();
  const T *pBias = bias.data();

  int rows = matrix.shape()[0];
  int cols = matrix.shape()[1];

#pragma omp parallel for collapse(2)
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {

      int idx = i * cols + j;

      pOut[idx] += pBias[j];
    }
  }

  return {output, matrix.shape()};
}

template <typename T>
Matrix<T> operator+(const std::vector<T> &bias, const Matrix<T> &left) {
  return left + bias;
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &matrix, const T &scalar) {

  std::vector<T> output = matrix.data();
  T *pOut = output.data();

  size_t size = matrix.size();

#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    pOut[i] += scalar;
  }

  return {output, matrix.shape()};
}

template <typename T>
Matrix<T> operator+(const T &scalar, const Matrix<T> &matrix) {
  return matrix + scalar;
}

template <typename T>
Matrix<T> operator-(const Matrix<T> &matrix, const T &scalar) {

  std::vector<T> output = matrix.data();
  T *pOut = output.data();

  size_t size = matrix.size();

#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    pOut[i] -= scalar;
  }

  return {output, matrix.shape()};
}

template <typename T>
Matrix<T> operator-(const T &scalar, const Matrix<T> &matrix) {
  return matrix - scalar;
}

template <typename T>
Matrix<T> operator-(const Matrix<T> &left, const Matrix<T> &right) {
  assert_eq(left.shape().size(), right.shape().size(),
            "Matrix::Operation::Dimension mismatch");
  assert_shape(left.shape(), right.shape(),
               "Matrix::Operation::Dimension mismatch");

  size_t size{left.size()};
  std::vector<T> diff(size);

  const T *pLeft = left.data().data();
  const T *pRight = right.data().data();
  T *pDiff = diff.data();

#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    pDiff[i] = pLeft[i] - pRight[i];
  }

  return {diff, left.shape()};
}

template <typename T>
Matrix<T> operator*(const T &scalar, const Matrix<T> &right) {

  size_t size{right.size()};
  std::vector<T> output(size);

  const T *pRight = right.data().data();
  T *pOut = output.data();

#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    pOut[i] = scalar * pRight[i];
  }

  return {output, right.shape()};
}

template <typename T>
Matrix<T> operator*(const Matrix<T> &right, const T &scalar) {
  return scalar * right;
}

// Element-wise Matrix multiplication.
template <typename T>
Matrix<T> operator*(const Matrix<T> &left, const Matrix<T> &right) {

  assert_shape(left.shape(), right.shape(),
               "Matrix::ElementWiseMult::Shapes must match exactly.");

  std::vector<T> out(left.size());
  T *pOut = out.data();
  const T *pLeft = left.data_ptr();
  const T *pRight = right.data_ptr();

#pragma omp simd
  for (size_t i = 0; i < left.size(); i++) {
    pOut[i] = pLeft[i] * pRight[i];
  }

  return {out, left.shape()};
}

template <typename T>
Matrix<T> operator/(const Matrix<T> &matrix, const T &scalar) {

  assert_ineq(scalar, (T)0, "Matrix::Zero division");

  std::vector<T> output = matrix.data();
  T *pOut = output.data();

  size_t size = matrix.size();
  T reciprocal = (T)1.0 / scalar;

#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    pOut[i] *= reciprocal;
  }

  return {output, matrix.shape()};
}

template <typename T>
Matrix<T> operator/(const T &scalar, const Matrix<T> &matrix) {

  std::vector<T> output = matrix.data();
  T *pOut = output.data();

  size_t size = matrix.size();

  for (size_t i = 0; i < size; i++) {

    assert_ineq(pOut[i], (T)0, "Matrix::Operation::ValueError::Zero Division");

    pOut[i] = scalar / pOut[i];
  }

  return {output, matrix.shape()};
}

// Element-wise division
template <typename T>
Matrix<T> operator/(const Matrix<T> &left, const Matrix<T> &right) {

  assert_shape(left.shape(), right.shape(),
               "Matrix::Division::ValueError::Dimensions mismatch");

  std::vector<T> output = left.data();
  T *pOut = output.data();
  const T *pRight = right.data_ptr();

  size_t size = left.size();

  for (size_t i = 0; i < size; i++) {

    assert_ineq(pRight[i], (T)0,
                "Matrix::Operation::ValueError::Zero Division");

    pOut[i] /= pRight[i];
  }

  return {output, left.shape()};
}
/********************************************************************************
 *
 * LinAlg Methods
 *
 *********************************************************************************/

template <typename T>
Matrix<T> &Matrix<T>::reshape(const std::vector<int> &new_shape) {
  size_t new_total_size = 1;
  for (int dim : new_shape) {
    assert_lineq(dim, 0, "Matrix::Reshape::Dimensions can't be negative.");
    new_total_size *= dim;
  }

  assert_eq(new_total_size, this->_size,
            "Matrix::Reshape::ValueError::Dimension mismatch");

  this->_shape = new_shape;

  return *this;
}

template <typename T> Matrix<T> &Matrix<T>::view(std::vector<int> new_shape) {
  int inferred_index = -1;
  size_t known_size = 1;

  for (size_t i = 0; i < new_shape.size(); ++i) {
    if (new_shape[i] == -1) {
      assert_eq(inferred_index, -1,
                "Matrix::Reshape::ValueError::Only one dimension can be -1.");
      inferred_index = i;
    } else {
      known_size *= new_shape[i];
    }
  }

  if (inferred_index != -1) {
    assert_eq(this->_size % (size_t)known_size, (size_t)0,
              "Matrix::Reshape::ValueError::Can't infer dimension");
    new_shape[inferred_index] = (int)(this->_size / known_size);
  }

  return reshape(new_shape);
}

/********************************************************************************
 *
 * Methods
 *
 *********************************************************************************/

template <typename T> T Math::Matrix<T>::at(size_t row, size_t col) {
  size_t ncols = (size_t)shape()[1];
  return _data[row * ncols + col];
}

// Get a requested row
template <typename T> Matrix<T> Matrix<T>::atRow(size_t row) const {
  size_t nrows = (size_t)_shape[0];
  size_t ncols = (size_t)_shape[1];

  Math::assert_lt(row, nrows, "Matrix::atRow");

  size_t start_idx = row * ncols;
  size_t end_idx = start_idx + ncols;

  std::vector<T> row_data(_data.begin() + start_idx, _data.begin() + end_idx);

  return Matrix<T>(std::move(row_data), {1, (int)ncols});
}

// Get a requested col
template <typename T> Matrix<T> Matrix<T>::atCol(size_t col) const {
  size_t nrows = (size_t)_shape[0];
  size_t ncols = (size_t)_shape[1];

  Math::assert_lt(col, ncols, "Matrix::atCol");

  std::vector<T> col_data;
  col_data.reserve(nrows);

  for (size_t i = 0; i < nrows; ++i) {
    col_data.push_back(_data[i * ncols + col]);
  }

  return Matrix<T>(std::move(col_data), {(int)nrows, 1});
}

} // namespace Math
