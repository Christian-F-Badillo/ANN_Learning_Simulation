#pragma once
#include "raylib.h"
#include "../math/matrix.h" // Necesario para acceder a los pesos
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

// Custom Types
using Topology = std::vector<int>;

enum class LayerType {
  Input,
  Hidden,
  Output,
};

struct NeuronTheme {
  Color inner;
  Color outer;
};

struct NetworkLayout {
  std::vector<std::vector<Vector2>> xy;
  float neuronRadius;
};

class DigitViewer {
public:
  DigitViewer();
  ~DigitViewer();
  void setData(const std::vector<int> &dataSample);
  void draw(Vector2 position, float rotation, float scale);

private:
  Texture2D texture;
};

inline DigitViewer::DigitViewer() { texture = {0}; }

inline DigitViewer::~DigitViewer() {
  if (texture.id != 0) {
    UnloadTexture(texture);
  }
}

inline void DigitViewer::setData(const std::vector<int> &dataSample) {
  if (texture.id != 0)
    UnloadTexture(texture);

  std::vector<unsigned char> pixelData;
  pixelData.reserve(64);

  for (int val : dataSample) {
    // Normalizamos 0-16 a 0-255
    int scaledVal = static_cast<int>((val / 16.0f) * 255.0f);
    pixelData.push_back(static_cast<unsigned char>(scaledVal));
  }

  Image sample = {.data = pixelData.data(),
                  .width = 8,
                  .height = 8,
                  .mipmaps = 1,
                  .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE};

  texture = LoadTextureFromImage(sample);
  SetTextureFilter(texture, TEXTURE_FILTER_POINT);
}

inline void DigitViewer::draw(Vector2 position, float rotation, float scale) {
  if (texture.id != 0) {
    DrawTextureEx(texture, position, rotation, scale, WHITE);
  }
}

inline NeuronTheme getLayerColors(LayerType type) {
  switch (type) {
  case LayerType::Input:
    return {{255, 100, 255, 200}, {120, 0, 120, 200}};
  case LayerType::Hidden:
    return {{100, 200, 255, 200}, {0, 40, 100, 200}};
  case LayerType::Output:
    return {{150, 255, 230, 200}, {0, 100, 80, 200}};
  default:
    return {{BLACK}, {RAYWHITE}};
  }
}

inline void drawFPSInfo(int fontSize, Color color) {
  DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, fontSize, color);
  DrawText(TextFormat("Frame time: %02.02f ms", GetFrameTime()), 60, 10,
           fontSize, color);
}

// Devuelve un par: Las posiciones y el radio ajustado
inline NetworkLayout calculateNetworkLayout(const Topology &topology,
                                            int screenWidth, int screenHeight,
                                            float targetRadius) {

  NetworkLayout layout;

  float finalRadius = targetRadius;
  float margin = 15.0f;

  int maxNeurons = 0;
  for (int n : topology)
    if (n > maxNeurons)
      maxNeurons = n;

  float diameter = targetRadius * 2;
  float totalHeightNeeded =
      (maxNeurons * diameter) + ((maxNeurons - 1) * margin);
  float availableHeight = screenHeight - 100.0f;

  if (totalHeightNeeded > availableHeight) {
    float spacePerNeuron = availableHeight / maxNeurons;

    if (diameter > spacePerNeuron) {
      finalRadius = (spacePerNeuron / 2.0f) - 2.0f;
      diameter = finalRadius * 2;
    }
    margin = spacePerNeuron - diameter;
  }

  int numLayers = topology.size();
  float layerStride = (float)screenWidth / (numLayers + 1);

  for (int i = 0; i < numLayers; i++) {
    std::vector<Vector2> layerPositions;
    int numNeurons = topology[i];
    float xPos = layerStride * (i + 1.75);

    float currentLayerHeight =
        (numNeurons * diameter) + ((numNeurons - 1) * margin);
    float startY = (screenHeight - currentLayerHeight) / 2.0f + finalRadius;

    for (int j = 0; j < numNeurons; j++) {
      float yPos = startY + j * (diameter + margin);
      layerPositions.push_back({xPos, yPos});
    }
    layout.xy.push_back(layerPositions);
  }

  layout.neuronRadius = finalRadius;
  return layout;
}

inline void drawNetwork(const NetworkLayout &layout) {
  int totalLayers = layout.xy.size();

  for (int i = 0; i < totalLayers; i++) {
    LayerType currentType;
    if (i == 0)
      currentType = LayerType::Input;
    else if (i == totalLayers - 1)
      currentType = LayerType::Output;
    else
      currentType = LayerType::Hidden;

    NeuronTheme theme = getLayerColors(currentType);

    for (const auto &pos : layout.xy[i]) {
      DrawCircleGradient((int)pos.x, (int)pos.y, layout.neuronRadius,
                         theme.inner, theme.outer);
      DrawCircleLines((int)pos.x, (int)pos.y, layout.neuronRadius, theme.outer);
    }
  }
}

template <typename T>
inline void drawNetworkConnections(
    const NetworkLayout &layout,
    const std::vector<std::shared_ptr<Math::Matrix<T>>> &params = {}) {

  int numLayers = layout.xy.size();
  float baseThickness = layout.neuronRadius * 0.15f;
  float radius = layout.neuronRadius;

  float visualScale = 20.0;

  bool useWeights = !params.empty();

  for (int layerId = 0; layerId < (numLayers - 1); layerId++) {

    const auto &layerIn = layout.xy[layerId];
    const auto &layerOut = layout.xy[layerId + 1];

    std::shared_ptr<Math::Matrix<T>> weightMatrix = nullptr;

    // Puntero a datos crudos y número de columnas para el cálculo de índice
    const std::vector<T> *wData = nullptr;
    int wCols = 0;

    if (useWeights && (layerId * 2) < params.size()) {
      weightMatrix = params[layerId * 2];
      if (weightMatrix) {
        // CORRECCIÓN: Usamos data() en lugar de at()
        wData = &weightMatrix->data();
        wCols =
            weightMatrix->shape()[1]; // Asumiendo shape() devuelve {rows, cols}
      }
    }

    for (size_t j = 0; j < layerIn.size(); j++) {
      Vector2 startPos = {layerIn[j].x + radius, layerIn[j].y};

      for (size_t k = 0; k < layerOut.size(); k++) {
        Vector2 endPos = {layerOut[k].x - radius, layerOut[k].y};

        Color lineColor = YELLOW;
        float alpha = 0.3f;
        float currentThickness = baseThickness;

        if (wData) {
          // Acceso directo al array plano: Fila * Columnas + Columna
          size_t idx = j * wCols + k;
          float val = 0.0f;

          // Check de seguridad
          if (idx < wData->size()) {
            val = (float)(*wData)[idx];
          }

          // Definición de colores: Azul (Positivo) / Rojo (Negativo)
          if (val > 0) {
            lineColor = (Color){0, 150, 255, 255}; // Azul Neón
          } else {
            lineColor = (Color){255, 50, 50, 255}; // Rojo Neón
          }

          // Cálculo de Opacidad
          float magnitude = std::abs(val) * visualScale;
          if (magnitude > 1.0f)
            magnitude = 1.0f;

          // SUELO DE OPACIDAD: Subido a 0.1 para que las conexiones débiles
          // sigan siendo visibles (estructura)
          if (magnitude < 0.1f)
            magnitude = 0.1f;

          alpha = magnitude;

          // Grosor Dinámico
          currentThickness = baseThickness * (0.5f + (magnitude * 1.5f));

          lineColor.a = (unsigned char)(alpha * 255.0f);
        } else {
          // Modo sin pesos (wireframe)
          lineColor = Fade(YELLOW, 0.15f);
        }

        // Dibujar solo si es mínimamente visible
        if (alpha > 0.01f) {
          DrawLineEx(startPos, endPos, currentThickness, lineColor);
        }
      }
    }
  }
}
