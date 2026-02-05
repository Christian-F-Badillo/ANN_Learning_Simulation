#pragma once
#include "raylib.h"
#include "../math/matrix.h"
#include <vector>
#include <cmath>
#include <memory>
#include <algorithm>

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

// Define the Topology type
using Topology = std::vector<int>;

// Define a Enum Class for each Layer type
enum class LayerType {
  Input,
  Hidden,
  Output,
};

// Data Struct to save Neuron Color
struct NeuronTheme {
  Color inner;
  Color outer;
};

// Data Struct for the Network Layout data.
struct NetworkLayout {
  std::vector<std::vector<Vector2>> xy;
  float neuronRadius;
};

// -----------------------------------------------------------------------------
// DigitViewer Class to render the test sample to infer.
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

// -----------------------------------------------------------------------------
// DigitViewer Class Methods
// -----------------------------------------------------------------------------

// Constructor
inline DigitViewer::DigitViewer() { texture = {0}; }

// Destructor
inline DigitViewer::~DigitViewer() {
  if (texture.id != 0) {
    UnloadTexture(texture);
  }
}

// Set Data to Viewer and render the 2D texture
inline void DigitViewer::setData(const std::vector<int> &dataSample) {
  if (texture.id != 0)
    UnloadTexture(texture);

  std::vector<unsigned char> pixelData;
  pixelData.reserve(64);

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

// Draw the current data
inline void DigitViewer::draw(Vector2 position, float rotation, float scale) {
  if (texture.id != 0) {
    DrawTextureEx(texture, position, rotation, scale, WHITE);
  }
}

// Auxiliar Functions to Draw Network

inline Color getLineColor(float val) {
  return (val > 0) ? Color({0, 150, 255, 255}) : Color({255, 50, 50, 255});
}

// -----------------------------------------------------------------------------
// Get the Color for each Layer Neuron
// -----------------------------------------------------------------------------

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

// Draw info abour current FPS and Frame Time
inline void drawFPSInfo(int fontSize, Color color) {
  DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, fontSize, color);
  DrawText(TextFormat("Frame time: %02.02f ms", GetFrameTime()), 80, 10,
           fontSize, color);
}

// -----------------------------------------------------------------------------
// Get the xy coords to draw each neuron based on screenWidth and screenHeight
// and Control UI Panel Width try set the set targetRadius if possible
// -----------------------------------------------------------------------------
inline NetworkLayout calculateNetworkLayout(const Topology &topology,
                                            int screenWidth, int screenHeight,
                                            float targetRadius,
                                            float panelWidth = 0.0f) {

  NetworkLayout layout;

  float finalRadius = targetRadius;
  float marginY = 15.0f;

  float startX = panelWidth + 60.0f;
  float usableWidth = (float)screenWidth - startX - 60.0f;

  int maxNeurons = 0;
  // Omitimos capa 0 (input colapsado) para calcular altura
  for (size_t i = 1; i < topology.size(); ++i) {
    if (topology[i] > maxNeurons)
      maxNeurons = topology[i];
  }
  if (maxNeurons == 0)
    maxNeurons = topology[0];

  float diameter = targetRadius * 2;
  float totalHeightNeeded =
      (maxNeurons * diameter) + ((maxNeurons - 1) * marginY);
  float availableHeight = screenHeight - 100.0f;

  if (totalHeightNeeded > availableHeight) {
    float spacePerNeuron = availableHeight / maxNeurons;
    if (diameter > spacePerNeuron) {
      finalRadius = (spacePerNeuron / 2.0f) - 2.0f;
      diameter = finalRadius * 2;
    }
    marginY = spacePerNeuron - diameter;
  }

  int numLayers = topology.size();

  float inputStride =
      120.0f; // Lees Space between input Layer and first Hidden Layer
  float hiddenStride = 0.0f;

  // Adaptative space between layer based on usableWidth.
  if (usableWidth < 300.0f)
    inputStride = usableWidth * 0.3f;

  // Divide the usable width to each layer
  if (numLayers > 2) {
    float remainingWidth = usableWidth - inputStride;
    hiddenStride = remainingWidth / (numLayers - 2);
  } else {
    inputStride = usableWidth;
    hiddenStride = 0.0f;
  }

  // Set the xy coords to each layer and neuron with Adaptative radius
  for (int i = 0; i < numLayers; i++) {
    std::vector<Vector2> layerPositions;
    int numNeurons = topology[i];

    float xPos = startX;
    if (i == 1) {
      xPos += inputStride;
    } else if (i > 1) {
      xPos += inputStride + ((i - 1) * hiddenStride);
    }

    float currentLayerHeight =
        (numNeurons * diameter) + ((numNeurons - 1) * marginY);
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
// Draw the Network Node by Node (neuron)
// -----------------------------------------------------------------------------
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

    if (i == 0) {
      if (!layout.xy[i].empty()) {
        float x = layout.xy[i][0].x;
        float minY = layout.xy[i].front().y;
        float maxY = layout.xy[i].back().y;
        float centerY = (minY + maxY) / 2.0f;

        DrawCircleGradient((int)x, (int)centerY, layout.neuronRadius,
                           theme.inner, theme.outer);
        DrawCircleLines((int)x, (int)centerY, layout.neuronRadius, theme.outer);

        DrawText("Input", (int)x - 20,
                 (int)centerY - (int)layout.neuronRadius - 20, 10, GRAY);
      }
      continue;
    }

    for (const auto &pos : layout.xy[i]) {
      DrawCircleGradient((int)pos.x, (int)pos.y, layout.neuronRadius,
                         theme.inner, theme.outer);
      DrawCircleLines((int)pos.x, (int)pos.y, layout.neuronRadius, theme.outer);
    }
  }
}

// --------------------------------------------------------------------------------------------------------------
// Draw the conections between Layers and set the color & alpha based on the
// currect weight between neurons
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
inline void drawNetworkConnections(
    const NetworkLayout &layout,
    const std::vector<std::shared_ptr<Math::Matrix<T>>> &params = {}) {

  int numLayers = layout.xy.size();
  float baseThickness = layout.neuronRadius * 0.05f;
  float radius = layout.neuronRadius;
  float visualScale = 100.0f;

  bool useWeights = !params.empty();

  for (int layerId = 0; layerId < (numLayers - 1); layerId++) {

    const auto &layerIn = layout.xy[layerId];
    const auto &layerOut = layout.xy[layerId + 1];

    const std::vector<T> *wData = nullptr;
    int wCols = 0;
    if (useWeights && (layerId * 2) < params.size()) {
      auto weightMatrix = params[layerId * 2];
      if (weightMatrix) {
        wData = &weightMatrix->data();
        wCols = weightMatrix->shape()[1];
      }
    }

    // Input -> First Hidden Layer
    if (layerId == 0) {
      float srcX = layerIn[0].x + radius;
      float srcY = (layerIn.front().y + layerIn.back().y) / 2.0f;
      Vector2 startPos = {srcX, srcY};

      for (size_t k = 0; k < layerOut.size(); k++) {
        Vector2 endPos = {layerOut[k].x - radius, layerOut[k].y};

        float avgVal = 0.0f;
        if (wData && !layerIn.empty()) {
          float sum = 0.0f;
          for (size_t j = 0; j < layerIn.size(); ++j) {
            size_t idx = j * wCols + k;
            if (idx < wData->size())
              sum += (float)(*wData)[idx];
          }
          avgVal = sum / (float)layerIn.size();
        }

        Color lineColor = YELLOW;
        float alpha = 0.3f;
        float currentThickness = baseThickness;

        if (wData) {
          lineColor = getLineColor(avgVal);

          float magnitude = std::abs(avgVal) * (visualScale * 3.0f);

          if (magnitude > 1.0f)
            magnitude = 1.0f;
          if (magnitude < 0.1f)
            magnitude = 0.1f;

          alpha = magnitude;
          currentThickness = baseThickness * (0.5f + (magnitude * 1.5f));
          lineColor.a = (unsigned char)(alpha * 255.0f);
        } else {
          lineColor = Fade(DARKGRAY, 0.3f);
        }
        DrawLineEx(startPos, endPos, currentThickness, lineColor);
      }
      continue;
    }

    // Rest of Layers
    for (size_t j = 0; j < layerIn.size(); j++) {
      Vector2 startPos = {layerIn[j].x + radius, layerIn[j].y};

      for (size_t k = 0; k < layerOut.size(); k++) {
        Vector2 endPos = {layerOut[k].x - radius, layerOut[k].y};

        Color lineColor = YELLOW;
        float alpha = 0.3f;
        float currentThickness = baseThickness;

        if (wData) {
          size_t idx = j * wCols + k;
          float val = (idx < wData->size()) ? (float)(*wData)[idx] : 0.0f;

          lineColor = getLineColor(val);

          float magnitude = std::abs(val) * visualScale;
          if (magnitude > 1.0f)
            magnitude = 1.0f;
          if (magnitude < 0.1f)
            magnitude = 0.1f;

          alpha = magnitude;
          currentThickness = baseThickness * (0.5f + (magnitude * 1.5f));
          lineColor.a = (unsigned char)(alpha * 255.0f);
        } else {
          lineColor = Fade(YELLOW, 0.15f);
        }

        if (alpha > 0.01f) {
          DrawLineEx(startPos, endPos, currentThickness, lineColor);
        }
      }
    }
  }
}
