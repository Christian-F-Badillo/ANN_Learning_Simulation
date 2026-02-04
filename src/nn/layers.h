#pragma once
#include "../math/matrix.h"
#include "../utils/asserts.h"
#include "ops.h"
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
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
  virtual Math::Matrix<T> forward(const Math::Matrix<T> &input);
  virtual Math::Matrix<T> backward(const Math::Matrix<T> &output_grad);
  virtual void _compute_param_grad(void);
  virtual void _get_params(void);

  std::vector<std::shared_ptr<Math::Matrix<T>>> params() { return params_; }
  std::vector<std::shared_ptr<Math::Matrix<T>>> param_grads() {
    return params_grad_;
  }

  virtual std::string get_type() const { return "Generic Layer"; }

  virtual std::string get_output_shape_str() const {
    if (output_) {
      auto shape = output_->shape();
      std::stringstream ss;
      ss << "(" << shape[0] << ", " << shape[1] << ")"; // (Batch, Neurons)
      return ss.str();
    }
    return "(None, " + std::to_string(neurons_) + ")";
  }

  virtual int get_total_params() const {
    int total = 0;
    for (const auto &p : params_) {
      total += p->size();
    }
    return total;
  }

  virtual void get_flat_layers(std::vector<Layer<T> *> &list) {
    list.push_back(this);
  }

  virtual std::map<std::string, std::shared_ptr<Math::Matrix<T>>>
  get_named_params() const {
    return {};
  }

  virtual std::map<std::string, std::shared_ptr<Math::Matrix<T>>>
  get_named_grads() const {
    return {};
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
 * Shared Methods Implementation
 *******************************************************/

// FORWARD
template <typename T>
Math::Matrix<T> Layer<T>::forward(const Math::Matrix<T> &input_data) {

  if (this->isFirst_) {
    this->_setup_layer(input_data);
    this->isFirst_ = false;
    this->_get_params();
  }

  this->input_ = std::make_shared<Math::Matrix<T>>(input_data);

  auto current_data = input_data;

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
template <typename T> void Layer<T>::_compute_param_grad(void) {

  this->params_grad_.clear();

  for (const auto &op : this->operations_) {

    auto paramOpType = std::dynamic_pointer_cast<Ops::ParamOperation<T>>(op);

    if (paramOpType) {
      this->params_grad_.push_back(paramOpType->param_grad());
    }
  }
}

// Get the params
template <typename T> void Layer<T>::_get_params(void) {
  this->params_.clear();
  this->params_grad_.clear();

  for (const auto &op : this->operations_) {
    auto paramOpType = std::dynamic_pointer_cast<Ops::ParamOperation<T>>(op);
    if (paramOpType) {
      this->params_.push_back(paramOpType->param());
      this->params_grad_.push_back(paramOpType->param_grad());
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

  std::string get_type() const override { return "Dense"; }

  std::map<std::string, std::shared_ptr<Math::Matrix<T>>>
  get_named_params() const override {
    std::map<std::string, std::shared_ptr<Math::Matrix<T>>> m;
    if (weights_ref_)
      m["weights"] = weights_ref_;
    if (bias_ref_)
      m["bias"] = bias_ref_;
    return m;
  }

  std::map<std::string, std::shared_ptr<Math::Matrix<T>>>
  get_named_grads() const override {
    std::map<std::string, std::shared_ptr<Math::Matrix<T>>> m;
    if (op_weights_)
      m["weights_grad"] = op_weights_->param_grad();
    if (op_bias_)
      m["bias_grad"] = op_bias_->param_grad();
    return m;
  }

private:
  std::shared_ptr<NN::Ops::Operation<T>> act_func_;
  std::shared_ptr<Math::Matrix<T>> weights_ref_;
  std::shared_ptr<Math::Matrix<T>> bias_ref_;
  std::shared_ptr<NN::Ops::WeightMultiply<T>> op_weights_;
  std::shared_ptr<NN::Ops::AddBias<T>> op_bias_;

  void _setup_layer(const Math::Matrix<T> &input) override;
};

// Implement Setup Layer
template <typename T>
void Dense<T>::_setup_layer(const Math::Matrix<T> &input) {

  std::random_device rd{};
  std::mt19937 gen{rd()};

  this->params_.clear();
  this->operations_.clear();

  int n_in = input.shape()[1];
  int n_out = this->neurons_;

  // Xavier Weight Initialization
  T n_avg = (T)(n_in + n_out);
  T std_dev = std::sqrt((T)2.0 / n_avg);

  std::normal_distribution<T> d{(T)0.0, std_dev};

  std::vector<T> dataWeights(n_in * n_out);
  std::vector<T> dataBias(n_out);

  for (auto &val : dataWeights)
    val = d(gen);
  for (auto &val : dataBias)
    val = (T)0.0;

  weights_ref_ = std::make_shared<Math::Matrix<T>>(
      dataWeights, std::vector<int>{n_in, n_out});
  bias_ref_ =
      std::make_shared<Math::Matrix<T>>(dataBias, std::vector<int>{1, n_out});

  auto ptr_weights = std::make_shared<Math::Matrix<T>>(
      dataWeights, std::vector<int>{n_in, n_out});
  auto ptr_bias =
      std::make_shared<Math::Matrix<T>>(dataBias, std::vector<int>{1, n_out});

  this->params_.push_back(ptr_weights);
  this->params_.push_back(ptr_bias);

  op_weights_ = std::make_shared<NN::Ops::WeightMultiply<T>>(weights_ref_);
  op_bias_ = std::make_shared<NN::Ops::AddBias<T>>(bias_ref_);

  this->operations_.push_back(
      std::make_shared<NN::Ops::WeightMultiply<T>>(ptr_weights));

  this->operations_.push_back(std::make_shared<NN::Ops::AddBias<T>>(ptr_bias));

  this->operations_.push_back(this->act_func_);
  this->_get_params();
}

/******************************************************************
 *
 * Implement a Sequential Class to generate a custom FeedForward Neural Network
 *
 *******************************************************************/

template <typename T> class Sequential : public Layer<T> {
public:
  Sequential(std::vector<std::shared_ptr<Layer<T>>> layers = {})
      : Layer<T>(0), layers_(layers) {}

  void add(std::shared_ptr<Layer<T>> layer);
  Math::Matrix<T> forward(const Math::Matrix<T> &input) override;
  Math::Matrix<T> backward(const Math::Matrix<T> &output_grad) override;
  void _compute_param_grad(void) override;
  void _get_params() override;

  std::string get_type() const override { return "Sequential"; }

  // Sequential delega la recolección a sus hijos
  void get_flat_layers(std::vector<Layer<T> *> &list) override {
    for (auto &layer : layers_) {
      layer->get_flat_layers(list);
    }
  }

protected:
  std::vector<std::shared_ptr<Layer<T>>> layers_;
  void _setup_layer(const Math::Matrix<T> &input) override {}
};

template <typename T> void Sequential<T>::add(std::shared_ptr<Layer<T>> layer) {
  layers_.push_back(layer);
}

template <typename T>
Math::Matrix<T> Sequential<T>::forward(const Math::Matrix<T> &input) {
  this->input_ = std::make_shared<Math::Matrix<T>>(input);

  Math::Matrix<T> current = input;

  for (auto &layer : layers_) {
    current = layer->forward(current);
  }

  if (this->isFirst_) {
    this->_get_params();
    this->isFirst_ = false;
  }

  this->output_ = std::make_shared<Math::Matrix<T>>(current);
  return current;
}

template <typename T>
Math::Matrix<T> Sequential<T>::backward(const Math::Matrix<T> &output_grad) {
  Math::assert_shape(this->output_->shape(), output_grad.shape(),
                     "Sequential Output mismatch");

  Math::Matrix<T> current_grad = output_grad;

  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    current_grad = (*it)->backward(current_grad);
  }

  this->inputGrad_ = std::make_shared<Math::Matrix<T>>(current_grad);

  this->_compute_param_grad();

  return current_grad;
}

template <typename T> void Sequential<T>::_compute_param_grad(void) {
  this->params_grad_.clear();
  for (auto &layer : layers_) {
    auto child_grads = layer->param_grads();
    this->params_grad_.insert(this->params_grad_.end(), child_grads.begin(),
                              child_grads.end());
  }
}

template <typename T> void Sequential<T>::_get_params() {
  this->params_.clear();
  this->params_grad_.clear();

  for (auto &layer : layers_) {
    layer->_get_params();
    auto child_params = layer->params();
    auto child_grads = layer->param_grads();

    this->params_.insert(this->params_.end(), child_params.begin(),
                         child_params.end());

    this->params_grad_.insert(this->params_grad_.end(), child_grads.begin(),
                              child_grads.end());
  }
}

} // namespace Layer
} // namespace NN
