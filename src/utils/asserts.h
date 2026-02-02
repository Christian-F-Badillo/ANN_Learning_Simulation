#pragma once
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Math {

// Util method to print a shape vector
inline std::string shape_to_string(const std::vector<int> &shape) {
  std::stringstream ss;
  ss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    ss << shape[i] << (i < shape.size() - 1 ? ", " : "");
  }
  ss << ")";
  return ss.str();
}

// Assert two shapes be equal
inline void assert_shape(const std::vector<int> &shape1,
                         const std::vector<int> &shape2,
                         const std::string &context = "") {
  if (shape1 != shape2) {
    std::string msg = "Shape Mismatch " + context + ": " +
                      shape_to_string(shape1) +
                      " != " + shape_to_string(shape2);
    throw std::invalid_argument(msg);
  }
}

// Assert two values be equal
template <typename T>
inline void assert_eq(const T &actual, const T &expected,
                      const std::string &context = "") {
  if (actual != expected) {
    std::stringstream ss;
    ss << "ValueError " << context << ": Expected " << expected << " but got "
       << actual;
    throw std::invalid_argument(ss.str());
  }
}

// Assert than a value is distinct than other
template <typename T>
inline void assert_ineq(const T &actual, const T &expected,
                        const std::string &context = "") {
  if (actual == expected) {
    std::stringstream ss;
    ss << "ValueError " << context << ": Expected !=" << expected << " but got "
       << actual;
    throw std::invalid_argument(ss.str());
  }
}

// Assert a value be less than a upper Bound
template <typename T>
inline void assert_lineq(const T &actual, const T &bound,
                         const std::string &context = "") {
  if (actual < bound) {
    std::stringstream ss;
    ss << "ValueError " << context << ": Expected > " << bound << " but got "
       << actual;
    throw std::invalid_argument(ss.str());
  }
}

// Assert a value be greater than a lower bound
template <typename T>
inline void assert_gineq(const T &actual, const T &bound,
                         const std::string &context = "") {
  if (actual > bound) {
    std::stringstream ss;
    ss << "ValueError " << context << ": Expected < " << bound << " but got "
       << actual;
    throw std::invalid_argument(ss.str());
  }
}

// Assert than a value es less than a limit
template <typename T>
inline void assert_lt(T index, T limit, const std::string &context = "") {
  if (index >= limit) {
    std::stringstream ss;
    ss << "IndexError " << context << ": Index " << index
       << " out of bounds (limit " << limit << ")";
    throw std::out_of_range(ss.str());
  }
}

// Assert than a Value is greater than a limit
template <typename T>
inline void assert_gt(T index, T limit, const std::string &context = "") {
  if (index <= limit) {
    std::stringstream ss;
    ss << "IndexError " << context << ": Index " << index
       << " out of bounds (limit " << limit << ")";
    throw std::out_of_range(ss.str());
  }
}

} // namespace Math
