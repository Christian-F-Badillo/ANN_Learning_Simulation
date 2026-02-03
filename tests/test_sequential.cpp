#include "../src/math/matrix.h"
#include "../src/nn/activation_func.h"
#include "../src/nn/layers.h"
#include "test_utils.h"
#include <iostream>
#include <memory>
#include <vector>

using namespace NN;
using namespace Math;

template <typename T> class MockLayer : public Layer::Layer<T> {
public:
  MockLayer() : Layer::Layer<T>(0) {}

  Matrix<T> forward(const Matrix<T> &input) override {
    // Output = Input * 2
    return input * (T)2.0;
  }

  Matrix<T> backward(const Matrix<T> &output_grad) override {
    // dL/dx = dL/dy * dy/dx = grad * 2
    return output_grad * (T)2.0;
  }

  // Implementaciones dummy para cumplir interfaz
  void _setup_layer(const Matrix<T> &input) override {}
  void _compute_param_grad(void) override {}
  void _get_params() override {}
};
int main() {
  std::cout << "=== TEST DE SEQUENTIAL LAYER ===" << std::endl;

  // ======================================================================
  // TEST 1: INTEGRACIÓN CON CAPAS DENSE (FORWARD)
  // Red: Input -> Dense(2 -> 3) -> ReLU -> Dense(3 -> 1) -> Sigmoid -> Output
  // ======================================================================
  TEST_CASE("Sequential: Integration with Dense Layers");

  auto model = std::make_shared<Layer::Sequential<float>>();

  // Capa 1: 2 inputs (implícito) -> 3 outputs
  auto act1 = std::make_shared<ActFunc::ReLU<float>>();
  model->add(std::make_shared<Layer::Dense<float>>(3, act1));

  // Capa 2: 3 inputs (implícito) -> 1 output
  auto act2 = std::make_shared<ActFunc::Sigmoid<float>>();
  model->add(std::make_shared<Layer::Dense<float>>(1, act2));

  // Input Batch: 4 muestras, 2 features
  Matrix<float> input({1, 1, 1, 1, 1, 1, 1, 1}, {4, 2});

  // Ejecutar Forward (Esto dispara el Lazy Init de todas las capas)
  Matrix<float> output = model->forward(input);

  // Verificación de Forma Final
  // Esperado: (BatchSize, FinalNeurons) -> (4, 1)
  ASSERT_EQ(output.shape()[0], 4);
  ASSERT_EQ(output.shape()[1], 1);

  std::cout << "   -> Shapes propagados correctamente." << std::endl;

  // ======================================================================
  // TEST 2: RECOLECCIÓN DE PARÁMETROS
  // El optimizador llamará a model->params() y espera recibir
  // los pesos de TODAS las capas en un solo vector plano.
  // ======================================================================
  TEST_CASE("Sequential: Parameter Collection");

  // Forzamos la actualización de la lista de parámetros
  model->_get_params();
  auto all_params = model->params();

  std::cout << "Model Params \n";
  for (const auto param : all_params) {
    std::cout << *param << "\n";
  }
  std::cout << '\n';
  // Análisis esperado:
  // Capa 1 (Dense): Tiene Pesos (W1) y Bias (B1) -> 2 tensores
  // Capa 2 (Dense): Tiene Pesos (W2) y Bias (B2) -> 2 tensores
  // Total = 4 tensores

  ASSERT_EQ(all_params.size(), (size_t)4);

  // Validar dimensiones de los tensores recolectados
  // Param 0 (W1): (In=2, Out=3)
  ASSERT_EQ(all_params[0]->shape()[0], 2);
  ASSERT_EQ(all_params[0]->shape()[1], 3);

  // Param 1 (B1): (1, 3)
  ASSERT_EQ(all_params[1]->shape()[1], 3);

  // Param 2 (W2): (In=3, Out=1) -> input viene de capa anterior
  ASSERT_EQ(all_params[2]->shape()[0], 3);
  ASSERT_EQ(all_params[2]->shape()[1], 1);

  // ======================================================================
  // TEST 3: BACKWARD PROPAGATION (Chain Rule)
  // Usamos MockLayers simples para verificar matemáticamente la cadena.
  // Red: Mock(x2) -> Mock(x2) -> Mock(x2)
  // Forward Total: x * 2 * 2 * 2 = 8x
  // Backward Total: grad * 2 * 2 * 2 = 8 * grad
  // ======================================================================
  TEST_CASE("Sequential: Backward Chaining logic");

  auto simple_seq = std::make_shared<Layer::Sequential<float>>();
  simple_seq->add(std::make_shared<MockLayer<float>>());
  simple_seq->add(std::make_shared<MockLayer<float>>());
  simple_seq->add(std::make_shared<MockLayer<float>>());

  Matrix<float> x(std::vector<float>({1.0f}), std::vector<int>({1, 1}));

  // Forward: 1.0 -> 2.0 -> 4.0 -> 8.0
  Matrix<float> y = simple_seq->forward(x);
  ASSERT_ALMOST_EQ(y.data_ptr()[0], 8.0f);

  // Backward: Gradiente inicial 1.0
  // 1.0 -> 2.0 -> 4.0 -> 8.0
  Matrix<float> dy(std::vector<float>({1.0f}), std::vector<int>({1, 1}));
  Matrix<float> dx = simple_seq->backward(dy);

  ASSERT_ALMOST_EQ(dx.data_ptr()[0], 8.0f);

  auto all_grads = simple_seq->param_grads();
  std::cout << "Model Param Grads \n";
  for (const auto param : all_grads) {
    std::cout << *param << "\n";
  }
  std::cout << '\n';

  return run_test_summary();
}
