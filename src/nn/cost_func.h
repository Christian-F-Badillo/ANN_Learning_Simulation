#pragma once
#include "../src/math/functions.h"
#include "../src/math/matrix.h"
#include "../src/math/matrix_linalg.h"
#include "../src/utils/asserts.h"
#include <memory>

namespace NN {
namespace CostFunc {

// =========================================================================
// BASE CLASS: Loss
// =========================================================================
template <typename T> class Loss {
public:
  virtual ~Loss() = default;

  T forward(const Math::Matrix<T> &prediction, const Math::Matrix<T> &target);

  Math::Matrix<T> backward();

protected:
  Loss() = default;

  std::shared_ptr<Math::Matrix<T>> prediction_;
  std::shared_ptr<Math::Matrix<T>> target_;
  std::shared_ptr<Math::Matrix<T>> diff_;

  virtual T _compute_loss_value() = 0;
  virtual Math::Matrix<T> _compute_input_grad() = 0;
};

template <typename T>
T Loss<T>::forward(const Math::Matrix<T> &prediction,
                   const Math::Matrix<T> &target) {
  Math::assert_shape(prediction.shape(), target.shape(), "Loss Forward");

  this->prediction_ = std::make_shared<Math::Matrix<T>>(prediction);
  this->target_ = std::make_shared<Math::Matrix<T>>(target);

  this->diff_ = std::make_shared<Math::Matrix<T>>(prediction - target);

  return this->_compute_loss_value();
}

template <typename T> Math::Matrix<T> Loss<T>::backward() {
  if (!this->diff_) {
    throw std::runtime_error("Loss::backward: Call forward first.");
  }
  return this->_compute_input_grad();
}

// =========================================================================
// MSE (Mean Squared Error)
// =========================================================================
template <typename T> class MeanSquareError : public Loss<T> {
public:
  T _compute_loss_value() override;
  Math::Matrix<T> _compute_input_grad() override;
};

template <typename T> T MeanSquareError<T>::_compute_loss_value() {

  auto squared_error = Math::Func::pow(*this->diff_, (T)2.0);

  auto sum_mat = Math::Linalg::sum(squared_error);

  T sum_val = sum_mat.data()[0];

  return sum_val / (T)this->prediction_->size();
}

template <typename T>
Math::Matrix<T> MeanSquareError<T>::_compute_input_grad() {

  T n = (T)this->prediction_->size();

  return (*this->diff_) * ((T)2.0 / n);
}

// =========================================================================
// Cross Entropy
// =========================================================================
template <typename T> class CategoricalCrossEntropy : public Loss<T> {
public:
  T _compute_loss_value() override;
  Math::Matrix<T> _compute_input_grad() override;
};

template <typename T> T CategoricalCrossEntropy<T>::_compute_loss_value() {
  const auto &y_pred = *this->prediction_;
  const auto &y_true = *this->target_;

  T eps = 1e-9;
  Math::Matrix<T> safe_pred = y_pred + eps;

  Math::Matrix<T> log_p = Math::Func::log(safe_pred);

  Math::Matrix<T> element_loss = y_true * log_p;

  Math::Matrix<T> sum_mat = Math::Linalg::sum(element_loss);
  T total_loss = sum_mat.data()[0];

  T N = (T)this->prediction_->shape()[0];

  return -total_loss / N;
}

template <typename T>
Math::Matrix<T> CategoricalCrossEntropy<T>::_compute_input_grad() {

  const auto &y_pred = *this->prediction_;
  const auto &y_true = *this->target_;
  T N = (T)y_pred.shape()[0];
  T eps = 1e-9;

  auto safe_pred = y_pred + eps;

  auto grad = (y_true / safe_pred) * (T)-1.0;

  return grad / N;
}

} // namespace CostFunc
} // namespace NN
