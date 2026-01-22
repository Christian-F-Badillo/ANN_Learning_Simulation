#include "../src/networkdraw.h"

NeuronTheme getLayerColors(Layer type) {
  switch (type) {
  case Layer::Input:
    return {{255, 100, 255, 200}, {120, 0, 120, 200}};

  case Layer::Hidden:
    return {{100, 200, 255, 200}, {0, 40, 100, 200}};

  case Layer::Output:
    return {{150, 255, 230, 200}, {0, 100, 80, 200}};

  default:
    return {{BLACK}, {RAYWHITE}};
  }
}

// Devuelve un par: Las posiciones y el radio ajustado
NetworkLayout calculateNetworkLayout(const Topology &topology, int screenWidth,
                                     int screenHeight, float targetRadius) {

  NetworkLayout layout;

  float finalRadius = targetRadius;
  float margin = 15.0f;

  // Buscamos la capa con más neuronas para ver el peor caso
  int maxNeurons = 0;
  for (int n : topology)
    if (n > maxNeurons)
      maxNeurons = n;

  float diameter = targetRadius * 2;
  float totalHeightNeeded =
      (maxNeurons * diameter) + ((maxNeurons - 1) * margin);
  float availableHeight = screenHeight - 80.0f; // Padding superior/inferior

  if (totalHeightNeeded > availableHeight) {
    float spacePerNeuron = availableHeight / maxNeurons;

    if (diameter > spacePerNeuron) {
      finalRadius = (spacePerNeuron / 2.0f) - 2.0f; // -2 de seguridad
      diameter = finalRadius * 2;
    }
    margin = spacePerNeuron - diameter;
  }

  // Generación de Coordenadas
  int numLayers = topology.size();
  float layerStride = (float)screenWidth / (numLayers + 1);

  for (int i = 0; i < numLayers; i++) {
    std::vector<Vector2> layerPositions;
    int numNeurons = topology[i];
    float xPos = layerStride * (i + 1);

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

void drawNetwork(const NetworkLayout &layout) {

  int totalLayers = layout.xy.size();

  for (int i = 0; i < totalLayers; i++) {

    // Lógica de Identificación de Capa
    Layer currentType;

    if (i == 0) {
      currentType = Layer::Input;
    } else if (i == totalLayers - 1) {
      currentType = Layer::Output;
    } else {
      currentType = Layer::Hidden;
    }

    // Recuperamos los colores correspondientes
    NeuronTheme theme = getLayerColors(currentType);

    // Renderizado de la Capa
    // Iteramos sobre todas las coordenadas pre-calculadas para esta capa
    for (const auto &pos : layout.xy[i]) {
      DrawCircleGradient((int)pos.x, (int)pos.y, layout.neuronRadius,
                         theme.inner, theme.outer);

      // Outer Ring
      DrawCircleLines((int)pos.x, (int)pos.y, layout.neuronRadius, theme.outer);
    }
  }
}

void drawNetworkConnections(const NetworkLayout &layout) {

  int numLayers = layout.xy.size();
  float thickness = layout.neuronRadius * 0.1f;
  float radius = layout.neuronRadius;

  for (int layerId = 0; layerId < (numLayers - 1); layerId++) {

    const auto &layerIn = layout.xy[layerId];
    const auto &layerOut = layout.xy[layerId + 1];

    for (const auto &neuronIn : layerIn) {

      Vector2 startPos = {neuronIn.x + radius, neuronIn.y};

      for (const auto &neuronOut : layerOut) {
        Vector2 endPos = {neuronOut.x - radius, neuronOut.y};
        DrawLineEx(startPos, endPos, thickness, YELLOW);
      }
    }
  }
}
