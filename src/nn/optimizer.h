#pragma once
#include "../math/functions.h"
#include "../math/matrix.h"
#include "utils/asserts.h"
#include <memory>
#include <vector>

namespace NN {
namespace Optimizer {

// Base Optimizer Class
template <typename T> class Optimizer {
public:
  ~Optimizer() = default;
  void setup(std::vector<std::shared_ptr<Math::Matrix<T>>> params,
             std::vector<std::shared_ptr<Math::Matrix<T>>> grads);
  virtual void step(void) = 0;

protected:
  Optimizer(T lr) : lr_(lr) { Math::assert_between(lr, (T)0, (T)1); }

  T lr_;
  std::vector<std::shared_ptr<Math::Matrix<T>>> params_;
  std::vector<std::shared_ptr<Math::Matrix<T>>> grads_;
};

// setup Method for base class Optimizer
template <typename T>
void Optimizer<T>::setup(std::vector<std::shared_ptr<Math::Matrix<T>>> params,
                         std::vector<std::shared_ptr<Math::Matrix<T>>> grads) {
  Math::assert_eq(params.size(), grads.size(),
                  "Optimizer::ValueError::Params and Grad size mismatch");
  this->params_ = params;
  this->grads_ = grads;
}

// SGD Optimizer
template <typename T> class SGD : public Optimizer<T> {
public:
  SGD(T learning_rate) : Optimizer<T>(learning_rate) {}

  void step() override;
};

// Step Method for SGD Optimizer
template <typename T> void SGD<T>::step(void) {
  for (size_t i = 0; i < this->params_.size(); i++) {
    Math::Matrix<T> &W = *this->params_[i];
    const Math::Matrix<T> &dW = *this->grads_[i];

    W = W - (dW * this->lr_);
  }
}

// ADAM Optimizer
template <typename T> class Adam : public Optimizer<T> {
public:
  Adam(T learning_rate, T beta1 = (T)0.9, T beta2 = (T)0.999,
       T epsilon = (T)1e-8)
      : Optimizer<T>(learning_rate), beta1_(beta1), beta2_(beta2),
        epsilon_(epsilon), t_(0) {
    Math::assert_between(beta1_, (T)0, (T)1);
    Math::assert_between(beta2_, (T)0, (T)1);
    Math::assert_gt(epsilon, (T)0);
  }

  void step() override;

private:
  T beta1_, beta2_, epsilon_; // Adam hyperparameters
  int t_;                     // time Step

  // Momentum history
  std::vector<std::shared_ptr<Math::Matrix<T>>> m_;
  std::vector<std::shared_ptr<Math::Matrix<T>>> v_;
};

// Adam Step Method
template <typename T> void Adam<T>::step(void) {

  if (m_.empty()) {
    for (const auto &param : this->params_) {
      m_.push_back(std::make_shared<Math::Matrix<T>>(
          std::vector<T>(param->size(), 0), param->shape()));
      v_.push_back(std::make_shared<Math::Matrix<T>>(
          std::vector<T>(param->size(), 0), param->shape()));
    }
  }

  t_++;

  for (size_t i = 0; i < this->params_.size(); i++) {

    auto &W = *this->params_[i];
    const auto &dW = *this->grads_[i];
    auto &m = *this->m_[i];
    auto &v = *this->v_[i];

    // Update Momentum
    m = (m * beta1_) + (dW * ((T)1.0 - beta1_));

    // update v
    auto dW_sq = Math::Func::pow(dW, (T)2.0);
    v = (v * beta2_) + (dW_sq * ((T)1.0 - beta2_));

    // Bias correction
    T beta1_t = std::pow(beta1_, (T)t_);
    T beta2_t = std::pow(beta2_, (T)t_);

    // Momentum update
    auto m_hat = m / ((T)1.0 - beta1_t);
    auto v_hat = v / ((T)1.0 - beta2_t);

    // Params update
    auto v_hat_sqrt = Math::Func::sqrt(v_hat);
    auto denominator = v_hat_sqrt + epsilon_;

    auto update = m_hat / denominator;
    W = W - (update * this->lr_);
  }
}

} // namespace Optimizer

} // namespace NN
