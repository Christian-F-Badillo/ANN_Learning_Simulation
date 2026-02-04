#pragma once
#include "raylib.h"
#include "../math/matrix.h" 
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

// -----------------------------------------------------------------------------
// TIPOS Y ESTRUCTURAS
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// CLASE DIGIT VIEWER (Visualizador de MNIST/Optdigits)
// -----------------------------------------------------------------------------
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

  // Normalizar datos de 0..16 a 0..255 para la textura
  for (int val : dataSample) {
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

// -----------------------------------------------------------------------------
// FUNCIONES DE ESTILO Y DIBUJO DE RED
// -----------------------------------------------------------------------------

inline NeuronTheme getLayerColors(LayerType type) {
  switch (type) {
  case LayerType::Input:
    return {{255, 100, 255, 200}, {120, 0, 120, 200}}; // Violeta
  case LayerType::Hidden:
    return {{100, 200, 255, 200}, {0, 40, 100, 200}};  // Azul Cielo
  case LayerType::Output:
    return {{150, 255, 230, 200}, {0, 100, 80, 200}};  // Verde Agua
  default:
    return {{BLACK}, {RAYWHITE}};
  }
}

inline void drawFPSInfo(int fontSize, Color color) {
  DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, fontSize, color);
  DrawText(TextFormat("Frame time: %02.02f ms", GetFrameTime()), 80, 10, fontSize, color);
}

// -----------------------------------------------------------------------------
// CÁLCULO DE LAYOUT (POSICIONES)
// -----------------------------------------------------------------------------
// panelWidth: Espacio reservado a la izquierda para la GUI
inline NetworkLayout calculateNetworkLayout(const Topology &topology,
                                            int screenWidth, int screenHeight,
                                            float targetRadius, 
                                            float panelWidth = 0.0f) {

  NetworkLayout layout;

  float finalRadius = targetRadius;
  float marginY = 15.0f; // Margen vertical entre neuronas

  // Calcular el área disponible real (restando el panel lateral)
  float startX = panelWidth + 40.0f; // Un poco de padding extra
  float usableWidth = (float)screenWidth - startX - 40.0f;

  // Determinar la capa más grande para ajustar la altura
  int maxNeurons = 0;
  for (int n : topology)
    if (n > maxNeurons) maxNeurons = n;

  float diameter = targetRadius * 2;
  float totalHeightNeeded = (maxNeurons * diameter) + ((maxNeurons - 1) * marginY);
  float availableHeight = screenHeight - 100.0f; 

  // Si no caben verticalmente, reducir el tamaño de las neuronas
  if (totalHeightNeeded > availableHeight) {
    float spacePerNeuron = availableHeight / maxNeurons;

    if (diameter > spacePerNeuron) {
      finalRadius = (spacePerNeuron / 2.0f) - 2.0f;
      diameter = finalRadius * 2;
    }
    marginY = spacePerNeuron - diameter;
  }

  int numLayers = topology.size();
  // Distribuir capas horizontalmente en el espacio disponible
  float layerStride = usableWidth / (numLayers > 1 ? numLayers - 1 : 1);

  for (int i = 0; i < numLayers; i++) {
    std::vector<Vector2> layerPositions;
    int numNeurons = topology[i];
    
    // Posición X basada en el offset del panel
    float xPos = startX + (i * layerStride);

    // Centrar verticalmente esta capa
    float currentLayerHeight = (numNeurons * diameter) + ((numNeurons - 1) * marginY);
    float startY = (screenHeight - currentLayerHeight) / 2.0f + finalRadius;

    for (int j = 0; j < numNeurons; j++) {
      float yPos = startY + j * (diameter + marginY);
      layerPositions.push_back({xPos, yPos});
    }
    layout.xy.push_back(layerPositions);
  }

  layout.neuronRadius = finalRadius;
  return layout;
}

// -----------------------------------------------------------------------------
// DIBUJADO DE NODOS (NEURONAS)
// -----------------------------------------------------------------------------
inline void drawNetwork(const NetworkLayout &layout) {
  int totalLayers = layout.xy.size();

  for (int i = 0; i < totalLayers; i++) {
    LayerType currentType;
    if (i == 0) currentType = LayerType::Input;
    else if (i == totalLayers - 1) currentType = LayerType::Output;
    else currentType = LayerType::Hidden;

    NeuronTheme theme = getLayerColors(currentType);

    for (const auto &pos : layout.xy[i]) {
      DrawCircleGradient((int)pos.x, (int)pos.y, layout.neuronRadius,
                         theme.inner, theme.outer);
      DrawCircleLines((int)pos.x, (int)pos.y, layout.neuronRadius, theme.outer);
    }
  }
}

// -----------------------------------------------------------------------------
// DIBUJADO DE CONEXIONES (PESOS) CON MAPA DE CALOR
// -----------------------------------------------------------------------------
template <typename T>
inline void drawNetworkConnections(
    const NetworkLayout &layout, 
    const std::vector<std::shared_ptr<Math::Matrix<T>>> &params = {}) {

  int numLayers = layout.xy.size();
  float baseThickness = layout.neuronRadius * 0.15f; 
  float radius = layout.neuronRadius;
  
  // Factor de sensibilidad visual:
  // Aumentar para que pesos pequeños (0.1, 0.05) se vean más brillantes.
  float visualScale = 12.0f; 

  bool useWeights = !params.empty();

  for (int layerId = 0; layerId < (numLayers - 1); layerId++) {

    const auto &layerIn = layout.xy[layerId];
    const auto &layerOut = layout.xy[layerId + 1];
    
    // Puntero a datos crudos y dimensiones para esta capa
    const std::vector<T>* wData = nullptr;
    int wCols = 0;

    // Buscamos la matriz de pesos correspondiente (ignorando biases)
    // params suele ser: [W0, b0, W1, b1 ...] -> Indices 0, 2, 4...
    if (useWeights && (layerId * 2) < params.size()) {
        auto weightMatrix = params[layerId * 2];
        if (weightMatrix) {
            wData = &weightMatrix->data();
            wCols = weightMatrix->shape()[1]; // Asumimos shape = {rows, cols}
        }
    }

    // Iterar Input Neurons (Origen)
    for (size_t j = 0; j < layerIn.size(); j++) {
      Vector2 startPos = {layerIn[j].x + radius, layerIn[j].y};

      // Iterar Output Neurons (Destino)
      for (size_t k = 0; k < layerOut.size(); k++) {
        Vector2 endPos = {layerOut[k].x - radius, layerOut[k].y};
        
        Color lineColor = YELLOW; 
        float alpha = 0.3f;       
        float currentThickness = baseThickness;

        if (wData) {
             // Cálculo de índice plano para Matrix (row-major o similar)
             size_t idx = j * wCols + k;
             float val = 0.0f;

             if (idx < wData->size()) {
                 val = (float)(*wData)[idx];
             }
             
             // Color: Azul (Positivo) / Rojo (Negativo)
             if (val > 0) {
                 lineColor = (Color){ 0, 150, 255, 255 }; // Azul Neón
             } else {
                 lineColor = (Color){ 255, 50, 50, 255 }; // Rojo Neón
             }
             
             // Opacidad basada en magnitud
             float magnitude = std::abs(val) * visualScale;
             if (magnitude > 1.0f) magnitude = 1.0f;
             
             // Suelo de visibilidad: Siempre mostrar la estructura débilmente
             if (magnitude < 0.1f) magnitude = 0.1f; 

             alpha = magnitude;
             
             // Grosor dinámico: Pesos fuertes son más gruesos
             currentThickness = baseThickness * (0.5f + (magnitude * 1.5f));

             lineColor.a = (unsigned char)(alpha * 255.0f);
        } else {
             // Modo Wireframe (si no hay pesos cargados)
             lineColor = Fade(YELLOW, 0.15f); 
        }

        // Dibujar solo si es mínimamente visible para ahorrar GPU
        if (alpha > 0.01f) {
            DrawLineEx(startPos, endPos, currentThickness, lineColor);
        }
      }
    }
  }
}