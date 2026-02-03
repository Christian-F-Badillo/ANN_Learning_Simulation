#pragma once
#include <iostream>
#include <limits>

namespace NN {
namespace Callbacks {

// =========================================================
// Enum Class to set the loss metric for EarlyStopping
// =========================================================
enum class Monitor { Train, Validation };

// =========================================================
// Class Base for Callbacks
// =========================================================
template <typename T> class Callback {
public:
  virtual ~Callback() = default;

  virtual void on_train_begin() {};

  virtual void on_epoch_end(int epoch, T train_loss, T val_loss,
                            bool &stop_training) {};

  virtual void on_train_end() {};
};

// =========================================================
// EarlyStopping
// =========================================================
template <typename T> class EarlyStopping : public Callback<T> {
public:
  EarlyStopping(Monitor mode = Monitor::Validation, int patience = 5,
                T min_delta = 0.0, bool verbose = true)
      : mode_(mode), patience_(patience), min_delta_(min_delta),
        verbose_(verbose) {
    wait_ = 0;
    best_loss_ = std::numeric_limits<T>::max();
    stopped_epoch_ = 0;
  }

  void on_train_begin() override {
    wait_ = 0;
    best_loss_ = std::numeric_limits<T>::max();
  }

  void on_epoch_end(int epoch, T train_loss, T val_loss,
                    bool &stop_training) override {
    T current_loss;
    if (mode_ == Monitor::Validation) {
      current_loss = val_loss;
    } else {
      current_loss = train_loss;
    }

    if (current_loss < (best_loss_ - min_delta_)) {
      best_loss_ = current_loss;
      wait_ = 0;
    } else {
      wait_++;

      if (verbose_ && (epoch % 10 == 0)) {
      }

      if (wait_ >= patience_) {
        stopped_epoch_ = epoch;
        stop_training = true;
        if (verbose_) {
          std::string metric_name =
              (mode_ == Monitor::Validation) ? "Validation" : "Train";
          std::cout << "\n[EarlyStopping] Train Stopped at epoch " << epoch
                    << ". Loss " << metric_name << " don't change in "
                    << patience_ << " epochs." << std::endl;
          std::cout << "[EarlyStopping] Best Loss (" << metric_name
                    << "): " << best_loss_ << std::endl;
        }
      }
    }
  }

private:
  Monitor mode_;
  int patience_;
  T min_delta_;
  bool verbose_;

  int wait_;
  T best_loss_;
  int stopped_epoch_;
};

} // namespace Callbacks
} // namespace NN
