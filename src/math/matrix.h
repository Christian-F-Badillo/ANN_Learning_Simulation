#pragma once
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <typeindex>
#include <vector>

template <typename T> class Matrix;

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &tensor);
template <typename T>
Matrix<T> operator+(const Matrix<T> &left, const Matrix<T> &right);
template <typename T>
Matrix<T> operator+(const Matrix<T> &left, const std::vector<T> &vector);
template <typename T>
Matrix<T> operator+(const std::vector<T> &vector, const Matrix<T> &left);
template <typename T>
Matrix<T> operator-(const Matrix<T> &left, const Matrix<T> &right);
template <typename T>
Matrix<T> operator*(const T &scalar, const Matrix<T> &tensor);
template <typename T>
Matrix<T> operator*(const Matrix<T> &right, const T &scalar);

template <typename T> class Matrix {

public:
  Matrix(std::vector<T> vectorIn, const std::vector<int> &shapeIn);
  Matrix(const std::vector<std::vector<T>> &matrix,
         const std::vector<int> &shapeIn);
  friend std::ostream &operator<< <>(std::ostream &os, const Matrix<T> &tensor);
  friend Matrix<T> operator+ <>(const Matrix<T> &left, const Matrix<T> &right);
  friend Matrix<T> operator+
      <>(const Matrix<T> &left, const std::vector<T> &bias);
  friend Matrix<T> operator+
      <>(const std::vector<T> &bias, const Matrix<T> &left);
  friend Matrix<T> operator- <>(const Matrix<T> &left, const Matrix<T> &right);
  friend Matrix<T> operator* <>(const T &scalar, const Matrix<T> &tensor);
  friend Matrix<T> operator* <>(const Matrix<T> &right, const T &scalar);

  const size_t &size() const;
  const std::vector<int> &shape() const;
  const std::vector<T> &data() const;
  const T *data_ptr() const;

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
  if (_shape.size() != 2) {
    throw std::invalid_argument("Matrix: dimension mismatch");
  } else if (_size != _data.size())
    throw std::invalid_argument(
        "Matrix: number of elements differs from shape.");
}
template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>> &matrix,
                  const std::vector<int> &shapeIn)
    : _data(_squeezeMatrix(matrix)), _shape(shapeIn), _size(_getSize(shapeIn)) {
  if (_shape.size() != 2) {
    throw std::invalid_argument("Matrix: dimension mismatch");
  } else if (_size != _data.size())
    throw std::invalid_argument(
        "Matrix: number of elements differs from shape.");
}

// ***************************************************************
// Utils Methods
// ***************************************************************
template <typename T>
size_t Matrix<T>::_getSize(const std::vector<int> &shapeIn) {
  size_t sizeInt{1};
  for (const auto &element : shapeIn) {
    if (element <= 0)
      throw std::invalid_argument(
          "Matrix: shape dimensions cannot be negative.");
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
std::ostream &operator<<(std::ostream &os, const Matrix<T> &tensor) {
  os << "Tensor(";
  if (tensor.data().empty()) {
    os << "[]";
  } else {
    size_t offset = 0;
    tensor.print_recursive(os, 0, offset, 0);
  }
  os << ", shape=(";
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    os << tensor.shape()[i] << (i < tensor.shape().size() - 1 ? "," : "");
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
Matrix<T> operator+(const Matrix<T> &left, const Matrix<T> &right) {
  if (left.shape().size() != right.shape().size()) {
    throw std::invalid_argument("Dimension mismatch");
  } else if (left.shape() != right.shape()) {
    throw std::invalid_argument("Dimension mismatch");
  }

  size_t size{left.size()};
  std::vector<T> sum(size);

  const T *pLeft = left.data().data();
  const T *pRight = right.data().data();
  T *pSum = sum.data();

#pragma omp simd
  for (size_t i = 0; i < size; i++) {
    pSum[i] = pLeft[i] + pRight[i];
  }

  return {sum, left.shape()};
}

template <typename T>
Matrix<T> operator+(const Matrix<T> &matrix, const std::vector<T> &bias) {

  if (bias.size() != (size_t)matrix.shape()[1]) {
    throw std::invalid_argument("Broadcast Add: El tamaño del vector no "
                                "coincide con las columnas (Bias mismatch).");
  }

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
Matrix<T> operator-(const Matrix<T> &left, const Matrix<T> &right) {
  if (left.shape().size() != right.shape().size()) {
    throw std::invalid_argument("Dimension mismatch");
  } else if (left.shape() != right.shape()) {
    throw std::invalid_argument("Dimension mismatch");
  }

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
