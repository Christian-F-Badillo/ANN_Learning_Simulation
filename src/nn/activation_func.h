#include "../math/functions.h"
#include "../math/matrix.h"
#include "ops.h"
#include <vector>

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

/*********************************************************************
 *
 * Implement a Linear (Identity) Activation Function
 *
 ********************************************************************/

template <typename T> class Linear : public Ops::Operation<T> {
public:
  Math::Matrix<T> _compute_output() override;
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override;
};

/*********************************************************************
 *
 * Foward and Backward for Linear
 *
 ********************************************************************/

template <typename T> Math::Matrix<T> Linear<T>::_compute_output() {
  return *this->input_;
}

template <typename T>
Math::Matrix<T>
Linear<T>::_compute_input_grad(const Math::Matrix<T> &output_grad) {
  return output_grad;
}

/*********************************************************************
 *
 * Implement a Softmax Activation Function
 *
 ********************************************************************/

template <typename T> class Softmax : public Ops::Operation<T> {
public:
  Math::Matrix<T> _compute_output() override;
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override;
};

/*********************************************************************
 *
 * Foward and Backward for Softmax
 *
 ********************************************************************/

template <typename T> Math::Matrix<T> Softmax<T>::_compute_output() {
  // Copia de entrada para no modificar la original
  Math::Matrix<T> result = *this->input_;

  int rows = result.shape()[0];
  int cols = result.shape()[1];
  const std::vector<T> &data = result.data();
  std::vector<T> out(data.size());
  T *pOut = out.data();

  for (int i = 0; i < rows; ++i) {
    int offset = i * cols;

    T max_val = data[offset];
    for (int j = 1; j < cols; ++j) {
      if (data[offset + j] > max_val)
        max_val = data[offset + j];
    }

    // 2. Exponencial y Suma
    T sum = (T)0;
    for (int j = 0; j < cols; ++j) {
      pOut[offset + j] = std::exp(data[offset + j] - max_val);
      sum += pOut[offset + j];
    }

    // 3. Normalización
    for (int j = 0; j < cols; ++j) {
      pOut[offset + j] /= sum;
    }
  }
  return {out, result.shape()};
}

template <typename T>
Math::Matrix<T>
Softmax<T>::_compute_input_grad(const Math::Matrix<T> &output_grad) {

  const Math::Matrix<T> &y = *this->output_;
  const std::vector<T> &y_data = y.data();
  const std::vector<T> &grad_data = output_grad.data();

  std::vector<T> input_grad_data(y_data.size());

  int rows = y.shape()[0];
  int cols = y.shape()[1];

  for (int i = 0; i < rows; ++i) {
    int offset = i * cols;

    T sum_y_grad = (T)0;
    for (int j = 0; j < cols; ++j) {
      sum_y_grad += y_data[offset + j] * grad_data[offset + j];
    }

    for (int j = 0; j < cols; ++j) {
      T y_val = y_data[offset + j];
      T grad_val = grad_data[offset + j];
      input_grad_data[offset + j] = y_val * (grad_val - sum_y_grad);
    }
  }

  return Math::Matrix<T>(input_grad_data, {rows, cols});
}

} // namespace ActFunc
} // namespace NN
