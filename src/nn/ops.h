#pragma once
#include "../src/math/matrix.h"
#include "../src/math/matrix_linalg.h"
#include <cstddef>
#include <memory>
#include <stdexcept>

/*********************************************************
 *
 * Define an Abstract Operation Base Class for common activation funcions
 *
 ***********************************************************/

template <typename T> class Operation {
public:
  virtual ~Operation<T>() = default;

  Matrix<T> forward(const Matrix<T> &input);
  Matrix<T> backward(const Matrix<T> &output_grad);

protected:
  Operation<T>() = default;
  std::shared_ptr<Matrix<T>> input_;
  std::shared_ptr<Matrix<T>> output_;
  std::shared_ptr<Matrix<T>> inputGrad_;

  virtual Matrix<T> _compute_output(void) = 0;
  virtual Matrix<T> _compute_input_grad(const Matrix<T> &output_grad) = 0;
};

/***************************
 *
 * Base Methods
 *
 ****************************/

// FORWARD

template <typename T> Matrix<T> Operation<T>::forward(const Matrix<T> &input) {

  this->input_ = std::make_shared<Matrix<T>>(input);
  this->output_ = std::make_shared<Matrix<T>>(this->_compute_output());

  return {this->output_->data(), this->output_->shape()};
}

// BACKWARD

template <typename T>
Matrix<T> Operation<T>::backward(const Matrix<T> &output_grad) {
  if (!this->input_) {
    throw std::runtime_error(
        "Operation::backward::Call backward before forward");
  }
  if (this->output_->shape() != output_grad.shape()) {
    throw std::invalid_argument(
        "Operation::backward::Dimension mistmatch at backward");
  }

  this->inputGrad_ =
      std::make_shared<Matrix<T>>(this->_compute_input_grad(output_grad));

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
  ParamOperation<T>(const Matrix<T> &param)
      : parameters(std::make_shared<Matrix<T>>(param)){};

  Matrix<T> backward(const Matrix<T> &output_grad);

protected:
  std::shared_ptr<Matrix<T>> parameters;
  std::shared_ptr<Matrix<T>> parameters_grad_;
  virtual Matrix<T> _compute_parameters_grad(const Matrix<T> &output_grad) = 0;
};

/******************************
 *
 * ParamOperation Methods
 *
 ******************************/

// BACKWARD
template <typename T>
Matrix<T> ParamOperation<T>::backward(const Matrix<T> &output_grad) {

  this->parameters_grad_ =
      std::make_shared<Matrix<T>>(this->_compute_parameters_grad(output_grad));

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

  WeightMultiply(const Matrix<T> &weights) : ParamOperation<T>(weights) {};

  Matrix<T> _compute_output(void) override;
  Matrix<T> _compute_input_grad(const Matrix<T> &output_grad) override;
  Matrix<T> _compute_parameters_grad(const Matrix<T> &output_grad) override;
};

/************************************************************************
 *
 * WeightMultiply Methods
 *
 *************************************************************************/

template <typename T> Matrix<T> WeightMultiply<T>::_compute_output() {

  return matmul(*this->input_, *this->parameters);
}

template <typename T>
Matrix<T> WeightMultiply<T>::_compute_input_grad(const Matrix<T> &output_grad) {

  return matmul(output_grad, transpose(*this->parameters));
}

template <typename T>
Matrix<T>
WeightMultiply<T>::_compute_parameters_grad(const Matrix<T> &output_grad) {

  return matmul(transpose(*this->input_), output_grad);
}

/***************************************************************************
 *
 * AddBias Class
 *
 ****************************************************************************/

template <typename T> class AddBias : public ParamOperation<T> {

  AddBias(const Matrix<T> &bias) : ParamOperation<T>(bias) {
    if (bias.shape()[0] != (size_t)1) {
      throw std::invalid_argument("AddBias::Bias nrows != 1");
    }
  };

  Matrix<T> _compute_output(void) override;
  Matrix<T> _compute_input_grad(const Matrix<T> &output_grad) override;
  Matrix<T> _compute_parameters_grad(const Matrix<T> &output_grad) override;
};

/************************************************************************
 *
 * WeightMultiply Methods
 *
 *************************************************************************/

template <typename T> Matrix<T> AddBias<T>::_compute_output() {

  return *this->input_ + *this->parameters;
}

template <typename T>
Matrix<T> AddBias<T>::_compute_input_grad(const Matrix<T> &output_grad) {

  return output_grad;
}

template <typename T>
Matrix<T> AddBias<T>::_compute_parameters_grad(const Matrix<T> &output_grad) {

  return 0; // Por Implementar
}
