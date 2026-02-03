#pragma once
#include "callbacks.h" // Incluimos los callbacks
#include "cost_func.h"
#include "layers.h"
#include "optimizer.h"
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace NN {

template <typename T> class Model {
public:
  Model() = default;

  void set_layers(std::shared_ptr<Layer::Layer<T>> network) {
    network_ = network;
  }

  void compile(std::shared_ptr<CostFunc::Loss<T>> loss,
               std::shared_ptr<Optimizer::Optimizer<T>> optimizer) {
    loss_ = loss;
    optimizer_ = optimizer;

    network_->_get_params();
    optimizer_->setup(network_->params(), network_->param_grads());
  }

  std::vector<std::shared_ptr<Math::Matrix<T>>> get_parameters() const {
    if (!network_)
      return {};
    return network_->params();
  }

  std::vector<std::shared_ptr<Math::Matrix<T>>> get_gradients() const {
    if (!network_)
      return {};
    return network_->param_grads();
  }

  // Print model Keras style
  void summary() const {
    if (!network_) {
      std::cout << "Model not initialized." << std::endl;
      return;
    }

    std::cout << "Model: \"Sequential_Neural_Network\"" << std::endl;
    std::cout
        << "_________________________________________________________________"
        << std::endl;
    std::cout << std::left << std::setw(25) << "Layer (type)" << std::setw(25)
              << "Output Shape" << std::setw(15) << "Param #" << std::endl;
    std::cout
        << "================================================================="
        << std::endl;

    std::vector<Layer::Layer<T> *> layers;
    network_->get_flat_layers(layers);

    int total_params = 0;
    int trainable_params = 0;

    for (size_t i = 0; i < layers.size(); ++i) {
      auto layer = layers[i];
      std::string name = layer->get_type() + "_" + std::to_string(i + 1);
      std::string shape = layer->get_output_shape_str();
      int params = layer->get_total_params();

      std::cout << std::left << std::setw(25) << name << std::setw(25) << shape
                << std::setw(15) << params << std::endl;

      total_params += params;
      trainable_params += params;
    }

    std::cout
        << "================================================================="
        << std::endl;
    std::cout << "Total params: " << total_params << std::endl;
    std::cout << "Trainable params: " << trainable_params << std::endl;
    std::cout << "Non-trainable params: 0" << std::endl;
    std::cout
        << "_________________________________________________________________"
        << std::endl;
  }

  std::vector<Layer::Layer<T> *> get_layers() const {
    std::vector<Layer::Layer<T> *> list;
    if (network_) {
      network_->get_flat_layers(list);
    }
    return list;
  }

  // Do a step in training
  T train_step(const Math::Matrix<T> &x_batch, const Math::Matrix<T> &y_batch) {
    if (!network_ || !loss_ || !optimizer_) {
      throw std::runtime_error("Model: Compile before training.");
    }

    auto predictions = network_->forward(x_batch);

    T current_loss = loss_->forward(predictions, y_batch);

    auto loss_grad = loss_->backward();
    network_->backward(loss_grad);

    optimizer_->setup(network_->params(), network_->param_grads());
    optimizer_->step();

    return current_loss;
  }
  // ===========================================================
  // FIT neither of Validation nor Callbacks
  // ===========================================================
  void fit(Math::Matrix<T> &x_train, Math::Matrix<T> &y_train, int epochs,
           int verbose = 10) {
    std::vector<std::shared_ptr<Callbacks::Callback<T>>> empty_callbacks;
    this->fit(x_train, y_train, epochs, empty_callbacks, verbose);
  }

  // ===========================================================
  // FIT with not Validation but Callbacks
  // ===========================================================
  void fit(Math::Matrix<T> &x_train, Math::Matrix<T> &y_train, int epochs,
           std::vector<std::shared_ptr<Callbacks::Callback<T>>> callbacks,
           int verbose = 10) {
    Math::Matrix<T> empty_val(std::vector<T>{}, std::vector<int>{0, 0});
    this->fit(x_train, y_train, empty_val, empty_val, epochs, callbacks,
              verbose);
  }

  // ===========================================================
  // FIT with Validation and Callbacks
  // ===========================================================
  void fit(Math::Matrix<T> &x_train, Math::Matrix<T> &y_train,
           Math::Matrix<T> &x_val, Math::Matrix<T> &y_val, int epochs,
           std::vector<std::shared_ptr<Callbacks::Callback<T>>> callbacks = {},
           int verbose = 10) {

    if (!network_ || !loss_ || !optimizer_) {
      throw std::runtime_error("Model: Compile before fitting.");
    }

    bool has_validation = (x_val.size() > 0 && y_val.size() > 0);
    bool stop_training = false;

    for (auto &cb : callbacks)
      cb->on_train_begin();

    std::cout << "Starting training for " << epochs << " epochs..."
              << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
      if (stop_training)
        break;

      auto predictions = network_->forward(x_train);
      T train_loss =
          loss_->forward(predictions, y_train); // Loss de entrenamiento

      auto loss_grad = loss_->backward();
      network_->backward(loss_grad);
      optimizer_->setup(network_->params(), network_->param_grads());
      optimizer_->step();

      // --- VALIDATION STEP ---
      T val_loss = (T)0.0;
      if (has_validation) {
        auto val_preds = network_->forward(x_val);
        val_loss = loss_->forward(val_preds, y_val);
      } else {
        val_loss = train_loss;
      }

      // --- CALLBACKS: ON_EPOCH_END ---
      for (auto &cb : callbacks) {
        cb->on_epoch_end(epoch, train_loss, val_loss, stop_training);
      }

      // --- LOGGING ---
      if (epoch % verbose == 0 || epoch == 1 || epoch == epochs) {
        std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                  << "Loss: " << train_loss;
        if (has_validation) {
          std::cout << " | Val Loss: " << val_loss;
        }
        std::cout << std::endl;
      }
    }

    // Llamar on_train_end
    for (auto &cb : callbacks)
      cb->on_train_end();

    if (!stop_training) {
      std::cout << "Training finished (completed all epochs)." << std::endl;
    }
  }

  Math::Matrix<T> predict(const Math::Matrix<T> &x) {
    return network_->forward(x);
  }

private:
  std::shared_ptr<Layer::Layer<T>> network_;
  std::shared_ptr<CostFunc::Loss<T>> loss_;
  std::shared_ptr<Optimizer::Optimizer<T>> optimizer_;
};

} // namespace NN
