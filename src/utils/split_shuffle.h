#pragma once

#include "../math/matrix.h"
#include <algorithm> // para std::shuffle
#include <memory>
#include <numeric> // para std::iota
#include <random>  // para std::mt19937
#include <stdexcept>
#include <vector>

namespace Utils {

// Estructura contenedora para devolver los 4 conjuntos de datos
template <typename T> struct TrainTestSplit {
  Math::Matrix<T> X_train;
  Math::Matrix<T> Y_train;
  Math::Matrix<T> X_val;
  Math::Matrix<T> Y_val;
};

class SplitShuffle {
public:
  /**
   * Divide y mezcla aleatoriamente los datos.
   * @param features Matriz de características [N x Features]
   * @param labels Matriz de etiquetas One-Hot [N x Outputs]
   * @param trainRatio Proporción para entrenamiento (ej. 0.8)
   */
  template <typename T>
  static TrainTestSplit<T> split(const Math::Matrix<T> &features,
                                 const Math::Matrix<T> &labels,
                                 float trainRatio = 0.8f, int seed = -1) {

    if (features.shape()[0] != labels.shape()[0]) {
      throw std::runtime_error(
          "SplitShuffle: Las filas de features y labels no coinciden.");
    }

    size_t totalRows = features.shape()[0];
    size_t featureCols = features.shape()[1];
    size_t labelCols = labels.shape()[1];

    // 1. Generar índices y mezclarlos (Shuffle)
    std::vector<size_t> indices(totalRows);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(seed == -1 ? rd() : seed);
    std::shuffle(indices.begin(), indices.end(), g);

    // 2. Calcular puntos de corte
    size_t trainCount = static_cast<size_t>(totalRows * trainRatio);
    size_t valCount = totalRows - trainCount;

    // 3. Preparar vectores
    std::vector<T> x_train, y_train, x_val, y_val;
    x_train.reserve(trainCount * featureCols);
    y_train.reserve(trainCount * labelCols);
    x_val.reserve(valCount * featureCols);
    y_val.reserve(valCount * labelCols);

    const std::vector<T> &src_x = features.data();
    const std::vector<T> &src_y = labels.data();

    // 4. Distribuir datos
    for (size_t i = 0; i < totalRows; ++i) {
      size_t rowIndex = indices[i];
      bool isTrain = (i < trainCount);

      auto &dest_x = isTrain ? x_train : x_val;
      auto &dest_y = isTrain ? y_train : y_val;

      // Copiar Features
      size_t startX = rowIndex * featureCols;
      dest_x.insert(dest_x.end(), src_x.begin() + startX,
                    src_x.begin() + startX + featureCols);

      // Copiar Labels
      size_t startY = rowIndex * labelCols;
      dest_y.insert(dest_y.end(), src_y.begin() + startY,
                    src_y.begin() + startY + labelCols);
    }

    return {
        Math::Matrix<T>(std::move(x_train),
                        {(int)trainCount, (int)featureCols}),
        Math::Matrix<T>(std::move(y_train), {(int)trainCount, (int)labelCols}),
        Math::Matrix<T>(std::move(x_val), {(int)valCount, (int)featureCols}),
        Math::Matrix<T>(std::move(y_val), {(int)valCount, (int)labelCols})};
  }
};

} // namespace Utils
