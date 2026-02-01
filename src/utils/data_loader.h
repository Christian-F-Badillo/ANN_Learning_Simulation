#pragma once

#include "../math/matrix.h"
#include <charconv>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace Data {

class DataLoader {

public:
  DataLoader(const std::string &fullPath) : path(fullPath) {};

  void loadData(void);

  // Ahora devolvemos referencias constantes a las Matrices
  // Usamos un getter seguro que valida si los datos fueron cargados
  const Math::Matrix<int> &getFeatures(void) const;
  const Math::Matrix<int> &getLabels(void) const;

private:
  // Usamos unique_ptr porque Matrix no tiene constructor por defecto
  // y queremos inicializarlas solo cuando tengamos los datos listos.
  std::unique_ptr<Math::Matrix<int>> features_mat;
  std::unique_ptr<Math::Matrix<int>> labels_mat;

  std::string path;
};

// -------------------------------------------------------------------------
// IMPLEMENTACIÓN
// -------------------------------------------------------------------------

inline void DataLoader::loadData(void) {
  std::ifstream file(path);
  if (!file.is_open())
    throw std::runtime_error("DataLoader::File not found: " + path);

  std::string line;

  // Vectores temporales planos para acumular datos
  std::vector<int> flat_features;
  std::vector<int> flat_labels;

  // Optimizacion: Reservar memoria si conocemos el tamaño aproximado (opcional)
  // flat_features.reserve(50000 * 784);

  int rows = 0;
  int cols = 0;

  while (std::getline(file, line)) {
    if (line.size() < 2)
      continue;

    std::string_view view(line);

    // Asumimos formato: feature1,feature2,...,label (o label al final)
    // Tu código original asumía label al final.

    // Separamos datos del label final
    // Nota: Ajusta esto si tu CSV tiene espacios o formato distinto
    std::string_view dataPart = view.substr(0, view.size() - 2);

    // Parseamos Label (último caracter)
    int label = view.back() - '0';
    flat_labels.push_back(label);

    // Parseamos Features
    size_t start = 0;
    size_t end = dataPart.find(',');
    int col_count = 0;

    while (end != std::string_view::npos) {
      std::string_view token = dataPart.substr(start, end - start);

      int value = 0;
      std::from_chars(token.data(), token.data() + token.size(), value);
      flat_features.push_back(value);
      col_count++;

      start = end + 1;
      end = dataPart.find(',', start);
    }

    // Último token antes del label
    std::string_view lastToken = dataPart.substr(start);
    int lastValue = 0;
    std::from_chars(lastToken.data(), lastToken.data() + lastToken.size(),
                    lastValue);
    flat_features.push_back(lastValue);
    col_count++;

    // Validación de consistencia de columnas
    if (rows == 0) {
      cols = col_count;
    } else if (col_count != cols) {
      throw std::runtime_error("DataLoader::Inconsistent column count at row " +
                               std::to_string(rows));
    }

    rows++;
  }

  // Construimos las matrices transfiriendo (move) los vectores planos
  // Features: [rows x cols]
  features_mat = std::make_unique<Math::Matrix<int>>(
      std::move(flat_features), std::vector<int>{rows, cols});

  // Labels: [rows x 1] (Vector columna) o [1 x rows] segun prefieras
  // Usualmente en NN se usa [rows x 1] para targets
  labels_mat = std::make_unique<Math::Matrix<int>>(std::move(flat_labels),
                                                   std::vector<int>{rows, 1});
}

inline const Math::Matrix<int> &DataLoader::getFeatures(void) const {
  if (!features_mat)
    throw std::runtime_error("DataLoader::Data not loaded!");
  return *features_mat;
}

inline const Math::Matrix<int> &DataLoader::getLabels(void) const {
  if (!labels_mat)
    throw std::runtime_error("DataLoader::Data not loaded!");
  return *labels_mat;
}

} // namespace Data
