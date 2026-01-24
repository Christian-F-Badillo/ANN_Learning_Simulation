#pragma once
#include <cstddef>
#include <iostream>
#include <vector>

template <typename T> class Tensor;

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);

template <typename T> class Tensor {

public:
  Tensor(std::vector<T> vectorIn, const std::vector<int> &shapeIn);
  Tensor(const std::vector<std::vector<T>> &matrix,
         const std::vector<int> &shapeIn);
  friend std::ostream &operator<< <>(std::ostream &os, const Tensor<T> &tensor);

private:
  std::vector<T> data;
  std::vector<int> shape;
  size_t size{};

  size_t _getSize(const std::vector<int> &shapeIn);
  std::vector<T> _squeezeMatrix(const std::vector<std::vector<T>> &matrix);
  void print_recursive(std::ostream &os, size_t dim_index, size_t &offset,
                       size_t indent_level) const;
};

/******************************************************
 *
 * Tensor Utility
 *
 *******************************************************/
