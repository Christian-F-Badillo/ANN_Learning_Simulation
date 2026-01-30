#include "../math/functions.h"
#include "../math/matrix.h"
#include "ops.h"

namespace NN {

// Create the Activation Function namespace into the general NN module
namespace ActFunc {

/*********************************************************************
 *
 * Implement a Sigmoid Activation Function
 *
 ********************************************************************/

template <typename T> class Sigmoid : public Ops::Operation<T> {
public:
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override;
  Math::Matrix<T> _compute_output(void) override;
};

/*********************************************************************
 *
 * Foward and Backward implementation for Sigmoid
 *
 ********************************************************************/

template <typename T> Math::Matrix<T> Sigmoid<T>::_compute_output() {

  return Math::Func::sigmoid(*this->input_);
}

template <typename T>
Math::Matrix<T>
Sigmoid<T>::_compute_input_grad(const Math::Matrix<T> &output_grad) {

  return ((*this->output_) * ((T)1.0 - (*this->output_))) * output_grad;
}

/*********************************************************************
 *
 * Implement a Tanh Activation Function
 *
 ********************************************************************/

template <typename T> class Tanh : public Ops::Operation<T> {
public:
  Math::Matrix<T> _compute_output() override;
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override;
};

/*********************************************************************
 *
 * Foward and Backward for ReLU
 *
 ********************************************************************/

template <typename T> Math::Matrix<T> Tanh<T>::_compute_output() {
  return Math::Func::tanh(*this->input_);
}

template <typename T>
Math::Matrix<T>
Tanh<T>::_compute_input_grad(const Math::Matrix<T> &output_grad) {

  Math::Matrix<T> y = *this->output_;
  Math::Matrix<T> local_deriv = (T)1.0 - (y * y);

  return output_grad * local_deriv;
}

/*********************************************************************
 *
 * Implement a ReLU (Rectified Linear Unit) Activation Function
 *
 ********************************************************************/

template <typename T> class ReLU : public Ops::Operation<T> {
public:
  Math::Matrix<T> _compute_output() override;
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override;
};

/*********************************************************************
 *
 * Foward and Backward for ReLU
 *
 ********************************************************************/

template <typename T> Math::Matrix<T> ReLU<T>::_compute_output() {
  return Math::Func::relu(*this->input_);
}

template <typename T>
Math::Matrix<T>
ReLU<T>::_compute_input_grad(const Math::Matrix<T> &output_grad) {

  Math::Matrix<T> mask = Math::Func::apply<T>(
      *this->input_, [](T x) { return x > (T)0 ? (T)1.0 : (T)0.0; });

  return output_grad * mask;
}

} // namespace ActFunc
} // namespace NN
