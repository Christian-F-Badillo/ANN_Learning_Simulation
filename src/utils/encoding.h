#pragma once
#include "../math/matrix.h"
#include <stdexcept>
#include <vector>

namespace Data {

class Encoder {
public:
  /**
   * Convierte un vector columna de etiquetas enteras en una Matriz One-Hot.
   * * @param labels: Matriz [N, 1] de enteros (obtenida de
   * DataLoader::getLabels)
   * @param numClasses: Número total de clases (ej. 10 para dígitos)
   * @return Math::Matrix<T> de dimensiones [N, numClasses] con 1.0 y 0.0
   */
  template <typename T>
  static Math::Matrix<T> toOneHot(const Math::Matrix<int> &labels,
                                  int numClasses) {

    if (labels.shape()[1] != 1) {
      throw std::runtime_error("Encoder::toOneHot: Labels matrix must be a "
                               "column vector [Rows x 1]");
    }

    int rows = labels.shape()[0];
    const std::vector<int> &rawLabels =
        labels.data(); // Accedemos al vector interno

    std::vector<T> oneHotData(rows * numClasses, (T)0.0);

    for (int i = 0; i < rows; ++i) {
      int label = rawLabels[i];

      if (label >= 0 && label < numClasses) {
        int index = (i * numClasses) + label;
        oneHotData[index] = (T)1.0;
      } else {
      }
    }

    return Math::Matrix<T>(std::move(oneHotData),
                           std::vector<int>{rows, numClasses});
  }

  /**
   * Inverso: Convierte probabilidades One-Hot de vuelta a etiquetas (ArgMax)
   * Útil para calcular Accuracy o mostrar predicciones en la GUI.
   */
  template <typename T>
  static int argMax(const std::vector<T> &probabilityVector) {
    int maxIndex = 0;
    T maxVal = probabilityVector[0];
    for (size_t i = 1; i < probabilityVector.size(); ++i) {
      if (probabilityVector[i] > maxVal) {
        maxVal = probabilityVector[i];
        maxIndex = (int)i;
      }
    }
    return maxIndex;
  }
};

} // namespace Data
