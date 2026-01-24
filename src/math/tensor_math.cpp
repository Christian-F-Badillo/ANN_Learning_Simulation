#include "tensor_math.h"
#include <cstddef>
#include <cstdlib>
#include <cxxabi.h>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>

// **************************************
// Constructors
// **************************************
template <typename T>
Tensor<T>::Tensor(std::vector<T> vectorIn, const std::vector<int> &shapeIn)
    : data(std::move(vectorIn)), shape(shapeIn), size(_getSize(shapeIn)) {
  if (size != data.size())
    throw std::invalid_argument("Number of elements differs from shape.");
}
template <typename T>
Tensor<T>::Tensor(const std::vector<std::vector<T>> &matrix,
                  const std::vector<int> &shapeIn)
    : data(_squeezeMatrix(matrix)), shape(shapeIn), size(_getSize(shapeIn)) {
  if (size != data.size())
    throw std::invalid_argument("Number of elements differs from shape.");
}

// ***************************************************************
// Utils Methods
// ***************************************************************
template <typename T>
size_t Tensor<T>::_getSize(const std::vector<int> &shapeIn) {
  size_t sizeInt{1};
  for (const auto &element : shapeIn) {
    if (element < 0)
      throw std::invalid_argument("Shape dimensions cannot be negative.");
    sizeInt *= (size_t)element;
  }
  return sizeInt;
}

template <typename T>
std::vector<T>
Tensor<T>::_squeezeMatrix(const std::vector<std::vector<T>> &matrix) {

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
void Tensor<T>::print_recursive(std::ostream &os, size_t dim_index,
                                size_t &offset, size_t indent_level) const {
  if (shape.empty())
    return;

  int current_dim_size = shape[dim_index];
  bool is_last_dim = (dim_index == shape.size() - 1);

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
      os << data[offset];
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
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
  os << "Tensor(";
  if (tensor.data.empty()) {
    os << "[]";
  } else {
    size_t offset = 0;
    tensor.print_recursive(os, 0, offset, 0);
  }
  os << ", shape=(";
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    os << tensor.shape[i] << (i < tensor.shape.size() - 1 ? "," : "");
  }
  os << "))";

  return os;
}

/*****************************************************
 *
 * Math Methods
 *
 ****************************************************/

/******************************************************
 *
 * Main
 *
 *****************************************************/

int main() {

  Tensor<int> tensor({1, 2, 3, 4, 5, 6}, {3, 2});

  std::cout << tensor << '\n';

  return 0;
}
