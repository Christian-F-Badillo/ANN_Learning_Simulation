#include "../src/math/matrix.h"
#include "../src/math/matrix_linalg.h"
#include "test_utils.h"
#include <iostream>

int main() {

  // ---------------------------------------------------------
  // CASO 1: Matrices Rectangulares (LA PRUEBA DE FUEGO)
  // (2x3) multiplicado por (3x2) -> Resultado debe ser (2x2)
  // Si tus bucles 'i, j, k' están mal, esto fallará o dará segfault.
  // ---------------------------------------------------------
  TEST_CASE("Matmul: Rectangular (2x3) * (3x2)");

  // A = [[1, 2, 3],
  //      [4, 5, 6]]
  Matrix<int> A({1, 2, 3, 4, 5, 6}, {2, 3});

  // B = [[7, 8],
  //      [9, 1],
  //      [2, 3]]
  Matrix<int> B({7, 8, 9, 1, 2, 3}, {3, 2});

  // Cálculo manual esperado:
  // [0][0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
  // [0][1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9  = 19
  // [1][0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
  // [1][1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
  Matrix<int> C = matmul(A, B);

  ASSERT_EQ(C.shape()[0], 2); // Filas
  ASSERT_EQ(C.shape()[1], 2); // Columnas

  const int *rawC = C.data_ptr();
  ASSERT_EQ(rawC[0], 31);
  ASSERT_EQ(rawC[1], 19);
  ASSERT_EQ(rawC[2], 85);
  ASSERT_EQ(rawC[3], 55);

  // ---------------------------------------------------------
  // CASO 2: Matriz Identidad (Propiedad Neutra)
  // A * I = A
  // ---------------------------------------------------------
  TEST_CASE("Matmul: Identity Matrix");

  Matrix<float> mat({1.5f, 2.5f, 3.5f, 4.5f}, {2, 2});
  Matrix<float> identity({1.0f, 0.0f, 0.0f, 1.0f}, {2, 2});

  Matrix<float> resId = matmul(mat, identity);

  const float *pMat = mat.data_ptr();
  const float *pResId = resId.data_ptr();

  for (size_t i = 0; i < mat.size(); i++) {
    ASSERT_ALMOST_EQ(pResId[i], pMat[i]);
  }

  // ---------------------------------------------------------
  // CASO 3: Dimensiones Incompatibles (Error Handling)
  // (2x2) * (3x2) -> Error, porque Cols A (2) != Filas B (3)
  // ---------------------------------------------------------
  TEST_CASE("Matmul: Invalid Dimensions Throw");

  Matrix<int> badA({1, 2, 3, 4}, {2, 2});
  Matrix<int> badB({1, 2, 3, 4, 5, 6}, {3, 2});

  ASSERT_THROWS(matmul(badA, badB), std::invalid_argument);

  // ---------------------------------------------------------
  // CASO 4: Vector Columna (Producto Punto simulado)
  // (1x3) * (3x1) -> Resultado escalar (1x1)
  // ---------------------------------------------------------
  TEST_CASE("Matmul: Dot Product Result (1x1)");

  Matrix<int> rowVec({1, 2, 3}, {1, 3});
  Matrix<int> colVec({4, 5, 6}, {3, 1});

  // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
  Matrix<int> scalarRes = matmul(rowVec, colVec);
  std::cout << scalarRes << '\n';
  ASSERT_EQ(scalarRes.shape()[0], 1);
  ASSERT_EQ(scalarRes.shape()[1], 1);
  ASSERT_EQ(scalarRes.data_ptr()[0], 32);

  TEST_CASE("Transpose: Square Matrix (2x2)");

  Matrix<int> square({1, 2, 3, 4}, {2, 2});

  Matrix<int> tSquare = transpose(square);

  // Verificamos dimensiones
  ASSERT_EQ(tSquare.shape()[0], 2);
  ASSERT_EQ(tSquare.shape()[1], 2);

  // Verificamos datos:
  // [1, 3]
  // [2, 4]
  const int *pSquare = tSquare.data_ptr();
  ASSERT_EQ(pSquare[0], 1);
  ASSERT_EQ(pSquare[1], 3); // Era row 1, col 0 -> ahora row 0, col 1
  ASSERT_EQ(pSquare[2], 2); // Era row 0, col 1 -> ahora row 1, col 0
  ASSERT_EQ(pSquare[3], 4);

  // ---------------------------------------------------------
  // CASO 2: Matriz Rectangular (2x3) -> (3x2)
  // ESTA ES LA PRUEBA CRÍTICA. Si tus bucles 'rows' y 'cols'
  // están invertidos o confundidos, esto fallará o dará SegFault.
  // ---------------------------------------------------------
  TEST_CASE("Transpose: Rectangular Matrix (2x3 -> 3x2)");

  // A = [[1, 2, 3],
  //      [4, 5, 6]]
  Matrix<int> rect({1, 2, 3, 4, 5, 6}, {2, 3});

  Matrix<int> tRect = transpose(rect);

  // Esperado: (3 filas, 2 columnas)
  // A_T = [[1, 4],
  //        [2, 5],
  //        [3, 6]]

  ASSERT_EQ(tRect.shape()[0], 3); // Filas nuevas = Columnas viejas
  ASSERT_EQ(tRect.shape()[1], 2); // Columnas nuevas = Filas viejas

  const int *pRect = tRect.data_ptr();

  // Fila 0
  ASSERT_EQ(pRect[0], 1);
  ASSERT_EQ(pRect[1], 4);
  // Fila 1
  ASSERT_EQ(pRect[2], 2);
  ASSERT_EQ(pRect[3], 5);
  // Fila 2
  ASSERT_EQ(pRect[4], 3);
  ASSERT_EQ(pRect[5], 6);

  // ---------------------------------------------------------
  // CASO 3: Vector Fila a Vector Columna (1x4) -> (4x1)
  // Crucial para los sesgos (biases) en la red neuronal.
  // ---------------------------------------------------------
  TEST_CASE("Transpose: Row Vector to Col Vector");

  Matrix<int> rowVec1({10, 20, 30, 40}, {1, 4});
  Matrix<int> colVec1 = transpose(rowVec1);

  ASSERT_EQ(colVec1.shape()[0], 4);
  ASSERT_EQ(colVec1.shape()[1], 1);

  const int *pCol = colVec1.data_ptr();
  ASSERT_EQ(pCol[0], 10);
  ASSERT_EQ(pCol[3], 40);

  // ---------------------------------------------------------
  // CASO 4: Identidad de Doble Transposición
  // (A^T)^T == A
  // ---------------------------------------------------------
  TEST_CASE("Transpose: Double Transpose Identity");

  Matrix<float> original({0.5f, -0.5f, 1.5f, 2.5f, -2.5f, 3.5f}, {2, 3});

  Matrix<float> doubleT = transpose(transpose(original));

  // Debe ser idéntica a la original
  ASSERT_EQ(doubleT.shape()[0], 2);
  ASSERT_EQ(doubleT.shape()[1], 3);

  const float *pOrig = original.data_ptr();
  const float *pDouble = doubleT.data_ptr();

  for (size_t i = 0; i < original.size(); i++) {
    ASSERT_ALMOST_EQ(pDouble[i], pOrig[i]);
  }

  // ---------------------------------------------------------
  // ESCENARIO 1: Suma de BIAS (Estándar en Redes Neuronales)
  // Broadcasting Horizontal: El vector se suma a cada FILA.
  // Requiere: vector.size() == matrix.columns()
  //
  // Matriz (2x3) + Vector (3)
  // [[10, 10, 10],   +  [1, 2, 3]  =  [[11, 12, 13],
  //  [20, 20, 20]]                    [21, 22, 23]]
  // ---------------------------------------------------------
  TEST_CASE("Broadcast: Suma de Bias (Column-wise)");

  Matrix<int> matBias({10, 10, 10, 20, 20, 20}, {2, 3});
  std::vector<int> vecBias = {1, 2, 3};

  try {
    Matrix<int> resBias = matBias + vecBias;

    // Si tu código es para Bias (pVec[j]), esto PASARÁ:
    const int *pRes = resBias.data_ptr();

    // Fila 0
    ASSERT_EQ(pRes[0], 11); // 10 + 1
    ASSERT_EQ(pRes[1], 12); // 10 + 2
    ASSERT_EQ(pRes[2], 13); // 10 + 3

    // Fila 1
    ASSERT_EQ(pRes[3], 21); // 20 + 1
    ASSERT_EQ(pRes[4], 22); // 20 + 2
    ASSERT_EQ(pRes[5], 23); // 20 + 3

    std::cout << "  -> Tu código implementa: SUMA DE BIAS (Correcto para NN)\n";

  } catch (...) {
    std::cout << "  -> Tu código NO implementa suma de columnas (o falló la "
                 "validación).\n";
  }

  // ---------------------------------------------------------
  // ESCENARIO 2: Suma por FILAS (Row-wise)
  // Broadcasting Vertical: El vector se suma a cada COLUMNA.
  // Requiere: vector.size() == matrix.rows()
  //
  // Matriz (2x3) + Vector (2)
  // [[1, 2, 3],   +  [100]  =  [[101, 102, 103],
  //  [4, 5, 6]]      [200]      [204, 205, 206]]
  // ---------------------------------------------------------
  TEST_CASE("Broadcast: Suma por Filas (Row-wise)");

  Matrix<int> matRow({1, 2, 3, 4, 5, 6}, {2, 3});
  std::vector<int> vecRow = {100, 200};

  try {
    Matrix<int> resRow = matRow + vecRow;

    // Si tu código es el que me mostraste antes (pVec[i]), esto PASARÁ:
    const int *pRes = resRow.data_ptr();

    // Fila 0 (se le suma 100)
    ASSERT_EQ(pRes[0], 101);
    ASSERT_EQ(pRes[1], 102);

    // Fila 1 (se le suma 200)
    ASSERT_EQ(pRes[3], 204);
    ASSERT_EQ(pRes[4], 205);

    std::cout << "  -> Tu código implementa: SUMA POR FILAS (Raro para NN "
                 "estándar)\n";

  } catch (...) {
    std::cout << "  -> Tu código NO implementa suma por filas.\n";
  }

  // ---------------------------------------------------------
  // ESCENARIO 3: Error de Dimensiones
  // Probamos un vector que no encaje ni con filas ni columnas
  // ---------------------------------------------------------
  TEST_CASE("Broadcast: Dimension Mismatch");

  Matrix<int> matErr({1, 2, 3, 4}, {2, 2});
  std::vector<int> vecErr = {1, 2, 3, 4, 5}; // Tamaño 5 (No es 2)

  // Debería lanzar excepción sin importar qué lógica uses
  ASSERT_THROWS(matErr + vecErr, std::invalid_argument);

  return run_test_summary();
}
