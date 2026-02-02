#include "../src/math/matrix.h"
#include "../src/nn/cost_func.h"
#include "test_utils.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace NN;
using namespace Math;

int main() {
  std::cout << "=== TEST SUITE: LOSS FUNCTIONS ===" << std::endl;

  // ======================================================================
  // TEST 1: Mean Squared Error (MSE)
  // Fórmula: Mean((y_pred - y_true)^2)
  // Gradiente: 2 * (y_pred - y_true) / N_elements
  // ======================================================================
  TEST_CASE("Mean Squared Error (MSE)");
  {
    // Configuración:
    // Predicciones: [[1, 2], [3, 4]]
    // Targets:      [[2, 2], [3, 6]]
    // Diferencia:   [[-1, 0], [0, -2]]
    // Cuadrado:     [[1, 0], [0, 4]]
    // Suma Total:   5
    // N_elementos:  4
    // Loss Esperado: 5 / 4 = 1.25

    Matrix<double> pred({1.0, 2.0, 3.0, 4.0}, {2, 2});

    Matrix<double> target({2.0, 2.0, 3.0, 6.0}, {2, 2});

    auto mse = std::make_shared<CostFunc::MeanSquareError<double>>();

    // 1. Forward Pass
    double loss = mse->forward(pred, target);
    ASSERT_ALMOST_EQ(loss, 1.25);

    // 2. Backward Pass (Gradiente)
    // Fórmula: 2 * (Pred - Target) / N
    // Diff: [[-1, 0], [0, -2]]
    // N = 4
    // Grad: 2 * Diff / 4 = Diff / 2 = [[-0.5, 0], [0, -1]]

    Matrix<double> grad = mse->backward();
    const double *pGrad = grad.data_ptr();

    ASSERT_ALMOST_EQ(pGrad[0], -0.5);
    ASSERT_ALMOST_EQ(pGrad[1], 0.0);
    ASSERT_ALMOST_EQ(pGrad[2], 0.0);
    ASSERT_ALMOST_EQ(pGrad[3], -1.0);
  }

  // ======================================================================
  // TEST 2: Categorical Cross Entropy (CCE)
  // Fórmula: -Sum(y_true * log(y_pred)) / BatchSize
  // Gradiente: -(y_true / y_pred) / BatchSize
  // ======================================================================
  TEST_CASE("Categorical Cross Entropy (CCE)");
  {
    // Configuración: Batch Size = 2, Clases = 2
    // Predicciones (Probabilidades):
    // [[0.9, 0.1],  (Muestra 1: Muy seguro de clase 0)
    //  [0.2, 0.8]]  (Muestra 2: Bastante seguro de clase 1)

    Matrix<double> pred({0.9, 0.1, 0.2, 0.8}, {2, 2});

    // Targets (One-Hot Encoding):
    // [[1, 0],  (Clase 0)
    //  [0, 1]]  (Clase 1)
    Matrix<double> target({1.0, 0.0, 0.0, 1.0}, {2, 2});

    auto cce = std::make_shared<CostFunc::CategoricalCrossEntropy<double>>();

    // 1. Forward Pass
    // Muestra 1: -1 * log(0.9) ≈ 0.10536
    // Muestra 2: -1 * log(0.8) ≈ 0.22314
    // Suma Total: 0.3285
    // Promedio (Batch=2): 0.16425

    double loss = cce->forward(pred, target);
    double expected_loss = -(std::log(0.9) + std::log(0.8)) / 2.0;

    ASSERT_ALMOST_EQ(loss, expected_loss);

    // 2. Backward Pass (Gradiente)
    // Fórmula: - (Target / Pred) / BatchSize
    // BatchSize = 2

    // Muestra 1:
    // dL/dp0 = -(1.0 / 0.9) / 2 = -0.5555...
    // dL/dp1 = -(0.0 / 0.1) / 2 = 0

    // Muestra 2:
    // dL/dp0 = -(0.0 / 0.2) / 2 = 0
    // dL/dp1 = -(1.0 / 0.8) / 2 = -0.625

    Matrix<double> grad = cce->backward();
    const double *pGrad = grad.data_ptr();

    ASSERT_ALMOST_EQ(pGrad[0], -1.0 / (0.9 * 2.0));
    ASSERT_ALMOST_EQ(pGrad[1], 0.0);
    ASSERT_ALMOST_EQ(pGrad[2], 0.0);
    ASSERT_ALMOST_EQ(pGrad[3], -1.0 / (0.8 * 2.0));
  }

  // ======================================================================
  // TEST 3: CCE Estabilidad Numérica (Log(0))
  // Verificamos que no explote si la predicción es 0
  // ======================================================================
  TEST_CASE("CCE: Numerical Stability (Epsilon check)");
  {
    // Predicción: 0.0 (Peligroso para log)
    // Target: 1.0 (Esto daría log(0) = -inf sin epsilon)
    Matrix<double> pred(std::vector<double>({0.0}), std::vector<int>({1, 1}));
    Matrix<double> target(std::vector<double>({1.0}), std::vector<int>({1, 1}));

    auto cce = std::make_shared<CostFunc::CategoricalCrossEntropy<double>>();

    // No debería lanzar excepción ni devolver NaN/Inf si implementaste epsilon
    double loss = cce->forward(pred, target);

    // Con epsilon 1e-9, log(1e-9) ≈ -20.7
    // El loss debería ser un número grande positivo, pero finito.
    bool is_finite = std::isfinite(loss);
    ASSERT_EQ(is_finite, true);

    // Gradiente tampoco debería ser NaN
    Matrix<double> grad = cce->backward();
    bool grad_finite = std::isfinite(grad.data_ptr()[0]);
    ASSERT_EQ(grad_finite, true);
  }

  return run_test_summary();
}
