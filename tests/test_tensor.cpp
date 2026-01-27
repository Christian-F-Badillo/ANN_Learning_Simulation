#include "../src/math/tensor_math.h"
#include "test_utils.h"

int main() {
  TEST_CASE("Creación de Tensores");
  Tensor<float> t({1, 2, 3, 4}, {2, 2});

  // Probar tamaños (size_t vs int)
  ASSERT_EQ(t.size(), (size_t)4);

  // Probar datos float
  ASSERT_ALMOST_EQ(t.data()[0], 1.0f);

  TEST_CASE("Validación de Errores");
  // Esto debería pasar exitosamente (porque esperamos que falle el código)
  ASSERT_THROWS(Tensor<float>({1, 2}, {5, 5}), std::invalid_argument);

  // Resumen final
  return run_test_summary();
}
