#include "../math/matrix.h"
#include "../utils/asserts.h"
#include "ops.h"
#include <cstddef>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace NN {

namespace Layer {

/********************************************************************************
 *
 * Layer Base Class for NN
 *
 ********************************************************************************/
template <typename T> class Layer {
public:
  virtual ~Layer() = default;
  Math::Matrix<T> forward(const Math::Matrix<T> &input);
  Math::Matrix<T> backward(const Math::Matrix<T> &output_grad);
  void _compute_param_grad(void);
  Math::Matrix<T> _get_params(void);
  std::vector<std::shared_ptr<Math::Matrix<T>>> params() { return params_; }
  std::vector<std::shared_ptr<Math::Matrix<T>>> param_grads() {
    return params_grad_;
  }

protected:
  Layer(int neurons) : neurons_(neurons), isFirst_(true) {};

  int neurons_;
  bool isFirst_;

  std::shared_ptr<Math::Matrix<T>> input_;
  std::shared_ptr<Math::Matrix<T>> inputGrad_;
  std::shared_ptr<Math::Matrix<T>> output_;

  std::vector<std::shared_ptr<NN::Ops::Operation<T>>> operations_;

  std::vector<std::shared_ptr<Math::Matrix<T>>> params_;
  std::vector<std::shared_ptr<Math::Matrix<T>>> params_grad_;

  virtual void _setup_layer(const Math::Matrix<T> &input) = 0;
};

/*******************************************************
 * Shared Methos
 *******************************************************/

// FORWARD
template <typename T>
Math::Matrix<T> Layer<T>::forward(const Math::Matrix<T> &input_data) {

  if (this->isFirst_) {
    this->_setup_layer(input_data);
    this->isFirst_ = false;
  }

  this->input_ = std::make_shared<Math::Matrix<T>>(input_data);

  Math::Matrix<T> current_data = input_data;

  for (auto &op : this->operations_) {
    current_data = op->forward(current_data);
  }

  this->output_ = std::make_shared<Math::Matrix<T>>(current_data);

  return current_data;
}

// BACKWARD
template <typename T>
Math::Matrix<T> Layer<T>::backward(const Math::Matrix<T> &output_grad) {

  Math::assert_shape(this->output_->shape(), output_grad.shape());

  Math::Matrix<T> current_output_grad = output_grad;

  for (auto i = this->operations_.rbegin(); i != this->operations_.rend();
       i++) {
    current_output_grad = (*i)->backward(current_output_grad);
  }

  this->inputGrad_ = std::make_shared<Math::Matrix<T>>(current_output_grad);

  this->_compute_param_grad();

  return current_output_grad;
}

// Backward on Parameters
template <typename T> void Layer<T>::_compute_param_grad() {

  this->params_grad_.clear();

  for (const auto &op : this->operations_) {

    auto paramOpType = std::dynamic_pointer_cast<Ops::ParamOperation<T>>(op);

    if (paramOpType) {
      this->params_grad_.push_back(paramOpType->param_grad());
    }
  }
}

// Get the params
template <typename T> Math::Matrix<T> Layer<T>::_get_params(void) {
  this->params_.clear();

  for (const auto &op : this->operations_) {

    auto paramOpType = std::dynamic_pointer_cast<Ops::ParamOperation<T>>(op);

    if (paramOpType) {
      this->params_grad_.push_back(paramOpType->param());
    }
  }
}

/***************************************************************************
 *
 * Class Layer Dense.
 *
 ***************************************************************************/

template <typename T> class Dense : public Layer<T> {
public:
  Dense(int neurons, std::shared_ptr<NN::Ops::Operation<T>> activation)
      : Layer<T>(neurons), act_func_(activation) {}

private:
  std::shared_ptr<NN::Ops::Operation<T>> act_func_;

  void _setup_layer(const Math::Matrix<T> &input) override;
};

// Implement Setup Layer
template <typename T>
void Dense<T>::_setup_layer(const Math::Matrix<T> &input) {

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<T> d{(T)0.0, (T)1.0};

  this->params_.clear();
  this->operations_.clear();

  int n_in = input.shape()[1];
  int n_out = this->neurons_;

  std::vector<T> dataWeights(n_in * n_out);
  std::vector<T> dataBias(n_out);

  for (auto &val : dataWeights)
    val = d(gen);
  for (auto &val : dataBias)
    val = d(gen);

  Math::Matrix<T> weights(std::move(dataWeights), {n_in, n_out});
  Math::Matrix<T> bias(std::move(dataBias), {1, n_out});

  auto ptr_weights = std::make_shared<Math::Matrix<T>>(std::move(weights));
  auto ptr_bias = std::make_shared<Math::Matrix<T>>(std::move(bias));

  this->params_.push_back(ptr_weights);
  this->params_.push_back(ptr_bias);

  this->operations_.push_back(
      std::make_shared<NN::Ops::WeightMultiply<T>>(*ptr_weights));

  this->operations_.push_back(std::make_shared<NN::Ops::AddBias<T>>(*ptr_bias));

  this->operations_.push_back(this->act_func_);
}

} // namespace Layer
} // namespace NN
