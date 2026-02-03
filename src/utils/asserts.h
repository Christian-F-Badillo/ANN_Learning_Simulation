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

// Assert than a value es less than an upper open bound
template <typename T>
inline void assert_lt(T value, T upper, const std::string &context = "") {
  if (value >= upper) {
    std::stringstream ss;
    ss << "ValueError " << context << ": Value " << value
       << " is greater than the upper bound " << upper << ")";
    throw std::out_of_range(ss.str());
  }
}

// Assert than a Value is greater than a lower open limit
template <typename T>
inline void assert_gt(T value, T lower, const std::string &context = "") {
  if (value <= lower) {
    std::stringstream ss;
    ss << "ValueError" << context << ": Value " << value
       << " is less than the lower bound" << lower << ")";
    throw std::out_of_range(ss.str());
  }
}

// Assert that a Value is between two open bouds
template <typename T>
inline void assert_between(T value, T lower, T upper,
                           const std::string &context = "") {
  if (value >= upper && value <= lower) {
    std::stringstream ss;
    ss << "ValueError" << context << ": Value " << value << " is not in ("
       << lower << ", " << upper << ")";
    throw std::out_of_range(ss.str());
  }
}

} // namespace Math
