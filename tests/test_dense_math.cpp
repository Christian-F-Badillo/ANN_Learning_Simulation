#include "../src/math/matrix.h"
#include "../src/nn/layers.h"
#include "../src/nn/ops.h"
#include "test_utils.h"
#include <iostream>
#include <memory>

using namespace NN;
using namespace Math;

// --- MOCK LINEAL (Identidad) ---
// Usamos esto para probar Dense sin que ReLU o Sigmoid alteren los números.
namespace NN {
namespace ActFunc {
template <typename T> class LinearMock : public Ops::Operation<T> {
public:
  Math::Matrix<T> _compute_output() override { return *this->input_; }
  Math::Matrix<T>
  _compute_input_grad(const Math::Matrix<T> &output_grad) override {
    return output_grad;
  }
};
} // namespace ActFunc
} // namespace NN

int main() {
  std::cout << "=== TEST DE VALIDACIÓN MATEMÁTICA (DENSE) ===" << std::endl;

  // CONFIGURACIÓN DEL ESCENARIO
  // Batch Size = 2
  // Input Features = 2
  // Output Neurons = 2

  // Matriz de Entrada (X)
  // [[1.0, 2.0],
  //  [3.0, 4.0]]
  Matrix<double> input({1.0, 2.0, 3.0, 4.0}, {2, 2});

  // 1. Instanciamos la capa
  auto linear_act = std::make_shared<ActFunc::LinearMock<double>>();
  Layer::Dense<double> dense(2, linear_act);

  // 2. Inicialización Perezosa (Dummy Run)
  // Ejecutamos una vez para que se asignen memoria a W y B
  dense.forward(input);

  // 3. INYECCIÓN DE PESOS CONTROLADOS
  // Obtenemos los punteros a los parámetros internos
  auto params = dense.params();
  // params[0] = Pesos (W), params[1] = Bias (B)

  // Definimos Pesos Conocidos (W)
  // [[0.1, 0.2],
  //  [0.3, 0.4]]
  Matrix<double> fixed_weights({0.1, 0.2, 0.3, 0.4}, {2, 2});

  // Definimos Bias Conocido (B)
  // [[0.5, 0.6]] (Vector fila broadcasting)
  Matrix<double> fixed_bias({0.5, 0.6}, {1, 2});

  // SOBREESCRIBIMOS LA MEMORIA DE LA CAPA
  // *params[0] accede al objeto real y usamos el operator= de Matrix (copia)
  *params[0] = fixed_weights;
  *params[1] = fixed_bias;

  std::cout << "-> Pesos y Bias inyectados manualmente." << std::endl;

  // ====================================================================
  // TEST 1: FORWARD PASS MATH
  // Fórmula: Y = (X * W) + B
  //
  // X * W:
  // [1, 2] . [[0.1, 0.2], [0.3, 0.4]] = [1*0.1 + 2*0.3, 1*0.2 + 2*0.4] =
  // [0.7, 1.0] [3, 4] . [[0.1, 0.2], [0.3, 0.4]] = [3*0.1 + 4*0.3, 3*0.2 +
  // 4*0.4] = [1.5, 2.2]
  //
  // + B ([0.5, 0.6]):
  // Row 1: [0.7+0.5, 1.0+0.6] = [1.2, 1.6]
  // Row 2: [1.5+0.5, 2.2+0.6] = [2.0, 2.8]
  // ====================================================================
  TEST_CASE("Math Check: Forward Calculation");

  Matrix<double> output = dense.forward(input);
  const double *pOut = output.data_ptr();

  // Verificamos valores exactos
  ASSERT_ALMOST_EQ(pOut[0], 1.2);
  ASSERT_ALMOST_EQ(pOut[1], 1.6);
  ASSERT_ALMOST_EQ(pOut[2], 2.0);
  ASSERT_ALMOST_EQ(pOut[3], 2.8);

  // ====================================================================
  // TEST 2: BACKWARD PASS MATH (INPUT GRADIENT)
  // Supongamos un gradiente de salida (dL/dY) simple de todo 1.0
  // [[1.0, 1.0],
  //  [1.0, 1.0]]
  //
  // Fórmula dL/dX = dL/dY * W^T
  // W^T = [[0.1, 0.3],
  //        [0.2, 0.4]]
  //
  // Cálculo (Fila 1): [1, 1] . W^T = [1*0.1 + 1*0.2, 1*0.3 + 1*0.4] = [0.3,
  // 0.7] Cálculo (Fila 2): Igual porque el gradiente de entrada es igual.
  // ====================================================================
  TEST_CASE("Math Check: Input Gradient (dL/dX)");

  Matrix<double> grad_output({1.0, 1.0, 1.0, 1.0}, {2, 2});

  Matrix<double> grad_input = dense.backward(grad_output);
  const double *pGradIn = grad_input.data_ptr();

  ASSERT_ALMOST_EQ(pGradIn[0], 0.3);
  ASSERT_ALMOST_EQ(pGradIn[1], 0.7);
  ASSERT_ALMOST_EQ(pGradIn[2], 0.3);
  ASSERT_ALMOST_EQ(pGradIn[3], 0.7);

  // ====================================================================
  // TEST 3: PARAMETER GRADIENTS MATH
  // Obtenemos los gradientes calculados internamente
  // ====================================================================
  auto param_grads = dense.param_grads();

  // --- 3A. Gradiente de Pesos (dW) ---
  // Fórmula dL/dW = X^T * dL/dY
  // X^T = [[1, 3],
  //        [2, 4]]
  // dL/dY = [[1, 1], [1, 1]]
  //
  // Fila 1: [1*1 + 3*1, 1*1 + 3*1] = [4, 4]
  // Fila 2: [2*1 + 4*1, 2*1 + 4*1] = [6, 6]
  TEST_CASE("Math Check: Weight Gradient (dL/dW)");
  const double *pGradW = param_grads[0]->data_ptr();

  ASSERT_ALMOST_EQ(pGradW[0], 4.0);
  ASSERT_ALMOST_EQ(pGradW[1], 4.0);
  ASSERT_ALMOST_EQ(pGradW[2], 6.0);
  ASSERT_ALMOST_EQ(pGradW[3], 6.0);

  // --- 3B. Gradiente de Bias (dB) ---
  // Fórmula dL/dB = Sum(dL/dY, axis=0) -> Suma vertical
  // Col 1: 1.0 + 1.0 = 2.0
  // Col 2: 1.0 + 1.0 = 2.0
  TEST_CASE("Math Check: Bias Gradient (dL/dB)");
  const double *pGradB = param_grads[1]->data_ptr();

  ASSERT_ALMOST_EQ(pGradB[0], 2.0);
  ASSERT_ALMOST_EQ(pGradB[1], 2.0);

  return run_test_summary();
}
