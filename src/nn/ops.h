#pragma once
#include "../src/math/matrix.h"
#include <cstddef>
#include <memory>
#include <stdexcept>

/*********************************************************
 *
 * Define an Abstract Operation Base Class for common activation funcions
 *
 ***********************************************************/

namespace NN {

namespace Ops {

template <typename T> class Operation {
public:
  virtual ~Operation<T>() = default;

  Math::Matrix<T> forward(const Math::Matrix<T> &input);
  Math::Matrix<T> backward(const Math::Matrix<T> &output_grad);

protected:
  Operation<T>() = default;
  std::shared_ptr<Math::Matrix<T>> input_;
  std::shared_ptr<Math::Matrix<T>> output_;
  std::shared_ptr<Math::Matrix<T>> inputGrad_;

  virtual Math::Matrix<T> _compute_output(void) = 0;
  virtual Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) = 0;
};

/***************************
 *
 * Base Methods
 *
 ****************************/

// FORWARD

template <typename T>
Math::Matrix<T> Operation<T>::forward(const Math::Matrix<T> &input) {

  this->input_ = std::make_shared<Math::Matrix<T>>(input);
  this->output_ = std::make_shared<Math::Matrix<T>>(this->_compute_output());

  return {this->output_->data(), this->output_->shape()};
}

// BACKWARD

template <typename T>
Math::Matrix<T> Operation<T>::backward(const Math::Matrix<T> &output_grad) {
  if (!this->input_) {
    throw std::runtime_error(
        "Operation::backward::Call backward before forward");
  }
  if (this->output_->shape() != output_grad.shape()) {
    throw std::invalid_argument(
        "Operation::backward::Dimension mistmatch at backward");
  }

  this->inputGrad_ =
      std::make_shared<Math::Matrix<T>>(this->_compute_input_grad(output_grad));

  if (this->inputGrad_->shape() != this->input_->shape()) {
    throw std::runtime_error(
        "Operation::backward::Dimension mistmatch: input shape != grad shape");
  }

  return {this->inputGrad_->data(), this->inputGrad_->shape()};
}

/***************************************************************************
 *
 * ParamOperation Class
 *
 ***************************************************************************/

template <typename T> class ParamOperation : public Operation<T> {

public:
  ParamOperation<T>(const Math::Matrix<T> &param)
      : parameters(std::make_shared<Math::Matrix<T>>(param)){};

  Math::Matrix<T> backward(const Math::Matrix<T> &output_grad);

protected:
  std::shared_ptr<Math::Matrix<T>> parameters;
  std::shared_ptr<Math::Matrix<T>> parameters_grad_;
  virtual Math::Matrix<T>
  _compute_parameters_grad(const Math::Matrix<T> &output_grad) = 0;
};

/******************************
 *
 * ParamOperation Methods
 *
 ******************************/

// BACKWARD
template <typename T>
Math::Matrix<T>
ParamOperation<T>::backward(const Math::Matrix<T> &output_grad) {

  this->parameters_grad_ = std::make_shared<Math::Matrix<T>>(
      this->_compute_parameters_grad(output_grad));

  if (this->parameters_grad_->shape() != this->parameters->shape()) {
    throw std::runtime_error(
        "ParamOperation::backward::Params grad shape mismatch");
  }

  return Operation<T>::backward(output_grad);
}

/***************************************************************************
 *
 * WeightMultiply Class
 *
 ****************************************************************************/

template <typename T> class WeightMultiply : public ParamOperation<T> {

public:
  WeightMultiply(const Math::Matrix<T> &weights)
      : ParamOperation<T>(weights) {};

  Math::Matrix<T> _compute_output(void) override;
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override;
  Math::Matrix<T>
  _compute_parameters_grad(const Math::Matrix<T> &output_grad) override;
};

/************************************************************************
 *
 * WeightMultiply Methods
 *
 *************************************************************************/

template <typename T> Math::Matrix<T> WeightMultiply<T>::_compute_output() {

  return matmul(*this->input_, *this->parameters);
}

template <typename T>
Math::Matrix<T>
WeightMultiply<T>::_compute_input_grad(const Math::Matrix<T> &output_grad) {

  return matmul(output_grad, transpose(*this->parameters));
}

template <typename T>
Math::Matrix<T> WeightMultiply<T>::_compute_parameters_grad(
    const Math::Matrix<T> &output_grad) {

  return matmul(transpose(*this->input_), output_grad);
}

/***************************************************************************
 *
 * AddBias Class
 *
 ****************************************************************************/

template <typename T> class AddBias : public ParamOperation<T> {

public:
  AddBias(const Math::Matrix<T> &bias) : ParamOperation<T>(bias) {
    if (bias.shape()[0] != (size_t)1) {
      throw std::invalid_argument("AddBias::Bias nrows != 1");
    }
  };

  Math::Matrix<T> _compute_output(void) override;
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override;
  Math::Matrix<T>
  _compute_parameters_grad(const Math::Matrix<T> &output_grad) override;
};

/************************************************************************
 *
 * WeightMultiply Methods
 *
 *************************************************************************/

template <typename T> Math::Matrix<T> AddBias<T>::_compute_output() {

  return *this->input_ + *this->parameters;
}

template <typename T>
Math::Matrix<T>
AddBias<T>::_compute_input_grad(const Math::Matrix<T> &output_grad) {

  return output_grad;
}

template <typename T>
Math::Matrix<T>
AddBias<T>::_compute_parameters_grad(const Math::Matrix<T> &output_grad) {

  return sum(output_grad, 0).reshape({1, output_grad.shape()[1]});
}

} // namespace Ops
} // namespace NN
