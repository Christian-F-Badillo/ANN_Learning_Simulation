#include "../src/math/matrix.h"
#include "../src/nn/callbacks.h"
#include "../src/nn/model.h"
#include "../src/nn/optimizer.h"
#include "test_utils.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace NN;
using namespace Math;

// --- UTILS PARA EL TEST ---

// 1. Activación Lineal
namespace NN {
namespace ActFunc {
template <typename T> class Linear : public Ops::Operation<T> {
public:
  Math::Matrix<T> _compute_output() override { return *this->input_; }
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override {
    return output_grad;
  }
};
} // namespace ActFunc
} // namespace NN

// 2. Callback Personalizado para contar épocas reales
// Nos sirve para verificar programáticamente en qué época se detuvo el modelo
template <typename T> class EpochCounter : public Callbacks::Callback<T> {
public:
  int total_epochs_run = 0;

  void on_train_begin() override { total_epochs_run = 0; }

  void on_epoch_end(int epoch, T t_loss, T v_loss, bool &stop) override {
    total_epochs_run = epoch;
  }
};

// ======================================================================
// MAIN TEST SUITE
// ======================================================================
int main() {
  std::cout << "=== TEST SUITE: CALLBACKS & ROBUST EARLY STOPPING ==="
            << std::endl;

  // Datos comunes: y = 2x
  Matrix<double> X({1, 2, 3, 4}, {4, 1});
  Matrix<double> Y({2, 4, 6, 8}, {4, 1});

  // Datos validación
  Matrix<double> X_val({1.5, 3.5}, {2, 1});
  Matrix<double> Y_val({3.0, 7.0}, {2, 1});

  // -----------------------------------------------------------------------
  // TEST 1: Early Stopping Clásico (Monitoreando Validación)
  // -----------------------------------------------------------------------
  TEST_CASE("Early Stopping con Monitor::Validation");
  {
    auto seq = std::make_shared<Layer::Sequential<double>>();
    seq->add(std::make_shared<Layer::Dense<double>>(
        1, std::make_shared<ActFunc::Linear<double>>()));

    Model<double> model;
    model.set_layers(seq);
    model.compile(std::make_shared<CostFunc::MeanSquareError<double>>(),
                  std::make_shared<Optimizer::Adam<double>>(0.01));

    // Configuración:
    // Max Epocas: 5000 (Exagerado)
    // Paciencia: 20 épocas
    // Monitor: Validation
    auto early_stop = std::make_shared<Callbacks::EarlyStopping<double>>(
        Callbacks::Monitor::Validation, 20, 1e-6, true);

    // Contador para verificar
    auto counter = std::make_shared<EpochCounter<double>>();

    std::vector<std::shared_ptr<Callbacks::Callback<double>>> cbs = {early_stop,
                                                                     counter};

    // Usamos el fit COMPLETO (con datos de validación)
    model.fit(X, Y, X_val, Y_val, 5000, cbs, 500);

    model.summary();

    std::cout << "   [VERIFICACION] Epocas ejecutadas: "
              << counter->total_epochs_run << " / 5000" << std::endl;

    if (counter->total_epochs_run >= 5000) {
      std::cerr << "   [FAIL] El entrenamiento no se detuvo anticipadamente."
                << std::endl;
      return 1;
    }

    // Verificar convergencia
    double final_loss =
        std::make_shared<CostFunc::MeanSquareError<double>>()->forward(
            model.predict(X_val), Y_val);
    if (final_loss > 0.1) {
      std::cerr << "   [FAIL] Se detuvo pero el loss es alto (" << final_loss
                << ")" << std::endl;
      return 1;
    }
    std::cout << "   [PASS] Se detuvo correctamente monitoreando validacion."
              << std::endl;
  }

  // -----------------------------------------------------------------------
  // TEST 2: Early Stopping "Robusto" (Sin Datos de Validación)
  // Monitoreando Monitor::Train
  // -----------------------------------------------------------------------
  TEST_CASE("Early Stopping Robusto (Monitor::Train, Sin Val Data)");
  {
    auto seq = std::make_shared<Layer::Sequential<double>>();
    seq->add(std::make_shared<Layer::Dense<double>>(
        1, std::make_shared<ActFunc::Linear<double>>()));

    Model<double> model;
    model.set_layers(seq);
    model.compile(std::make_shared<CostFunc::MeanSquareError<double>>(),
                  std::make_shared<Optimizer::Adam<double>>(0.01));

    // Configuración:
    // Monitor: Train (Porque no le pasaremos datos de validación)
    auto early_stop = std::make_shared<Callbacks::EarlyStopping<double>>(
        Callbacks::Monitor::Train, 30, 1e-6, true);

    auto counter = std::make_shared<EpochCounter<double>>();
    std::vector<std::shared_ptr<Callbacks::Callback<double>>> cbs = {early_stop,
                                                                     counter};

    std::cout << "   [INFO] Probando sobrecarga de fit() sin matrices de "
                 "validacion..."
              << std::endl;

    // Usamos el fit INTERMEDIO (Sin X_val, Y_val)
    // Esto prueba que tu Model.h maneja bien la creación de matrices vacías
    // internas
    model.fit(X, Y, 5000, cbs, 500);

    std::cout << "   [VERIFICACION] Epocas ejecutadas: "
              << counter->total_epochs_run << " / 5000" << std::endl;

    if (counter->total_epochs_run >= 5000) {
      std::cerr << "   [FAIL] El entrenamiento no se detuvo." << std::endl;
      return 1;
    }

    std::cout << "   [PASS] Se detuvo correctamente monitoreando el Train Set."
              << std::endl;
  }

  // -----------------------------------------------------------------------
  // TEST 3: Caso Borde - Error de Usuario
  // Usuario pide Monitor::Validation pero NO pasa datos de validación.
  // El código debe ser robusto y usar Train Loss como proxy (según tu Model.h)
  // o fallar elegantemente. Según nuestra implementación en Model.h: "val_loss
  // = train_loss" si no hay datos.
  // -----------------------------------------------------------------------
  TEST_CASE("Robustez: Monitor::Validation solicitado SIN datos de validacion");
  {
    auto seq = std::make_shared<Layer::Sequential<double>>();
    seq->add(std::make_shared<Layer::Dense<double>>(
        1, std::make_shared<ActFunc::Linear<double>>()));
    Model<double> model;
    model.set_layers(seq);
    model.compile(std::make_shared<CostFunc::MeanSquareError<double>>(),
                  std::make_shared<Optimizer::Adam<double>>(0.01));

    // El usuario pide validar, pero...
    auto early_stop = std::make_shared<Callbacks::EarlyStopping<double>>(
        Callbacks::Monitor::Validation, 20, 1e-6, true);
    auto counter = std::make_shared<EpochCounter<double>>();

    // ... NO pasa datos de validación.
    // El modelo debería asignar internamente val_loss = train_loss, permitiendo
    // que EarlyStopping funcione.
    model.fit(X, Y, 5000, {early_stop, counter}, 500);

    if (counter->total_epochs_run >= 5000) {
      std::cerr << "   [FAIL] Fallback de robustez falló." << std::endl;
      return 1;
    }
    std::cout << "   [PASS] El sistema hizo fallback seguro (usando Train Loss "
                 "como proxy)."
              << std::endl;
  }

  return run_test_summary();
}
