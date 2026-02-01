#include "../src/math/matrix.h"
#include "../src/nn/activation_func.h"
#include "../src/nn/layers.h"
#include "test_utils.h"
#include <memory>

using namespace NN;
using namespace Math;

int main() {

  // ======================================================================
  // TEST 1: Inicialización y Shapes (Forward)
  // Verificamos que la "Inicialización Perezosa" (Lazy Init) funcione
  // y cree los pesos la primera vez que ve datos.
  // ======================================================================
  TEST_CASE("Dense: Initialization & Forward Shapes");
  {
    // Batch de 2 muestras, con 3 features cada una
    Matrix<float> input({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, {2, 3});

    // Creamos la capa: 4 neuronas de salida, activación ReLU
    auto relu = std::make_shared<NN::ActFunc::ReLU<float>>();

    // Instanciamos Dense (Input size no se define aquí, se infiere del forward)
    // Constructor: Dense(int neurons, shared_ptr<Operation> activation)
    Layer::Dense<float> denseLayer(4, relu);

    // --- FORWARD ---
    Matrix<float> output = denseLayer.forward(input);

    // Verificación de Shapes de Salida
    // Esperado: (BatchSize, Neuronas) -> (2, 4)
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 4);

    // Verificación de Parámetros Internos (Pesos y Bias)
    // Debemos llamar a _get_params() o acceder si son públicos para verificar
    // Asumiendo que Layer tiene el método público params() que devuelve el
    // vector
    auto params = denseLayer.params();

    // Debe haber 2 matrices de parámetros: [0]=Pesos, [1]=Bias
    ASSERT_EQ(params.size(), (size_t)2);

    // Chequeo de Pesos (W): (InputFeatures, Neurons) -> (3, 4)
    ASSERT_EQ(params[0]->shape()[0], 3);
    ASSERT_EQ(params[0]->shape()[1], 4);

    // Chequeo de Bias (B): (1, Neurons) -> (1, 4)
    ASSERT_EQ(params[1]->shape()[0], 1);
    ASSERT_EQ(params[1]->shape()[1], 4);
  }

  // ======================================================================
  // TEST 2: Backward Pass y Gradientes
  // Verificamos que las derivadas fluyan correctamente y generen
  // gradientes para los pesos.
  // ======================================================================
  TEST_CASE("Dense: Backward Flow & Parameter Gradients");
  {
    // Configuración:
    // Input: (2 samples, 5 features)
    // Dense: 3 neuronas de salida
    Matrix<double> input({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 5});

    auto relu = std::make_shared<NN::ActFunc::ReLU<double>>();
    Layer::Dense<double> denseLayer(3, relu);

    // 1. Forward (Obligatorio antes de backward)
    denseLayer.forward(input);

    // 2. Simular un Gradiente que viene de la siguiente capa
    // Debe tener la misma forma que el output del forward: (Batch, Neuronas) ->
    // (2, 3)
    Matrix<double> output_grad({0.5, -0.5, 1.0, 0.0, 0.5, 0.5}, {2, 3});

    // 3. Backward
    Matrix<double> input_grad = denseLayer.backward(output_grad);

    // --- Verificación del Gradiente de Entrada (dX) ---
    // Debe tener la misma forma que el input original: (2, 5)
    ASSERT_EQ(input_grad.shape()[0], 2);
    ASSERT_EQ(input_grad.shape()[1], 5);

    // --- Verificación de Gradientes de Parámetros (dW, dB) ---
    // La capa debe haber calculado y guardado los gradientes internamente
    auto param_grads = denseLayer.param_grads();

    ASSERT_EQ(param_grads.size(), (size_t)2);

    // Gradiente de Pesos (dL/dW): Debe coincidir con W -> (5, 3)
    ASSERT_EQ(param_grads[0]->shape()[0], 5);
    ASSERT_EQ(param_grads[0]->shape()[1], 3);

    // Gradiente de Bias (dL/dB): Debe coincidir con B -> (1, 3)
    ASSERT_EQ(param_grads[1]->shape()[0], 1);
    ASSERT_EQ(param_grads[1]->shape()[1], 3);
  }

  return run_test_summary();
}
