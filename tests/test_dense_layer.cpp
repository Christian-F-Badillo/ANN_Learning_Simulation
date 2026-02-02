#include "../src/math/matrix.h"
#include "../src/nn/layers.h"
#include "../src/nn/ops.h" // Asegúrate que WeightMultiply y AddBias están aquí
#include "test_utils.h"
#include <iostream>
#include <memory>

// --- MOCK ACTIVATION ---
// Una activación "dummy" que no hace nada (Identidad)
// para probar la capa Dense sin ruido de otras fórmulas.
namespace NN {
namespace ActFunc {
template <typename T> class LinearMock : public Ops::Operation<T> {
public:
  Math::Matrix<T> _compute_output() override {
    // Simplemente devuelve la entrada tal cual (f(x) = x)
    // Nota: Asegúrate de desreferenciar input_ correctamente
    return *this->input_;
  }
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override {
    // La derivada de x es 1, así que devuelve el gradiente tal cual
    return output_grad;
  }
};
} // namespace ActFunc
} // namespace NN

using namespace NN;
using namespace Math;

int main() {
  std::cout << "=== INICIANDO TEST DE DENSE LAYER ===" << std::endl;

  // DATA SETUP
  // Batch Size: 2, Features: 3
  Matrix<float> input({1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, {2, 3});

  // Layer SETUP
  int n_out = 2; // Out shape
  auto activation = std::make_shared<ActFunc::LinearMock<float>>();

  // Assert not null pointer
  if (!activation) {
    std::cerr << "FATAL: Error al crear activación mock." << std::endl;
    return -1;
  }

  TEST_CASE("Dense Layer Instantiation");

  // Dense(neuronas, activacion)
  Layer::Dense<float> dense(n_out, activation);
  std::cout << "   -> Capa instanciada correctamente." << std::endl;

  // FORWARD PASS
  TEST_CASE("Dense Forward Pass (Lazy Init)");
  Matrix<float> output = dense.forward(input);

  // Assert output shape
  // Input (2,3) * Weights (3,2) = Output (2,2)
  ASSERT_EQ(output.shape()[0], 2);
  ASSERT_EQ(output.shape()[1], n_out);

  // Assert correct pointer creation
  auto params = dense.params();
  ASSERT_EQ(params.size(), (size_t)2); // Weights & Bias

  if (params[0] == nullptr || params[1] == nullptr) {
    std::cerr << "   [FAIL] Los parámetros se crearon como punteros nulos!"
              << std::endl;
    return 1;
  }
  std::cout << "   -> Forward completado. Shapes correctos." << std::endl;

  // BACKWARD PASS
  TEST_CASE("Dense Backward Pass");

  // Artificial Grad
  Matrix<float> grad_output({0.5, 0.5, 0.5, 0.5}, {2, 2});

  Matrix<float> grad_input = dense.backward(grad_output);

  // Assert input grad shape
  ASSERT_EQ(grad_input.shape()[0], 2);
  ASSERT_EQ(grad_input.shape()[1], 3);
  std::cout << "   -> Backward completado. Shapes correctos." << std::endl;

  // Parameter Grads
  TEST_CASE("Parameter Gradients Check");
  auto grads = dense.param_grads();

  ASSERT_EQ(grads.size(), (size_t)2);

  // Assert pointer to grads exits
  if (!grads[0]) {
    std::cerr << "   [FAIL] El gradiente de los PESOS es null." << std::endl;
    return 1;
  }
  if (!grads[1]) {
    std::cerr << "   [FAIL] El gradiente del BIAS es null." << std::endl;
    return 1;
  }

  // Assert dimensions
  // Grad W should be (3, 2)
  ASSERT_EQ(grads[0]->shape()[0], 3);
  ASSERT_EQ(grads[0]->shape()[1], 2);

  // Grad B should be (1, 2)
  ASSERT_EQ(grads[1]->shape()[0], 1);
  ASSERT_EQ(grads[1]->shape()[1], 2);

  return run_test_summary();
}
