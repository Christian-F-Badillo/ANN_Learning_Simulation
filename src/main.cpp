#include "raylib.h"
#include <vector>

// Macros
#define WINDOW_HEIGHT 1000
#define WINDOW_WIDTH 1500
#define FPS 60

using Topology = std::vector<int>;

enum class Layer {
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

void drawFPSInfo(int fontSize, Color color);
void switchFullScreen();
void drawNetwork(const NetworkLayout &layout);
NetworkLayout calculateNetworkLayout(const Topology &topology, int screenWidth,
                                     int screenHeight, float targetRadius);
NeuronTheme getLayerColors(Layer type);

int main(int argc, char *argv[]) {

  // Window Config
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
  // Create window and Opengl Context
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
             "Simulation of Artificial Neural Networks");

  // Set Frame per Second

  float radius{50.0f};
  Topology topology{20, 20, 10, 6, 5};
  NetworkLayout layout = calculateNetworkLayout(topology, GetScreenWidth(),
                                                GetScreenHeight(), radius);

  // Main app running
  while (!WindowShouldClose()) {

    switchFullScreen();

    if (IsWindowResized()) {
      layout = calculateNetworkLayout(topology, GetScreenWidth(),
                                      GetScreenHeight(), radius);
    }

    BeginDrawing();
    ClearBackground(BLACK);

    drawFPSInfo(10, GREEN);
    drawNetwork(layout);

    EndDrawing();
  }

  CloseWindow();

  return 0;
}

/*******************************************************************************************************************
 *
 * CUSTOM Functions
 *
 ********************************************************************************************************************/

// Print FPS Info
void drawFPSInfo(int fontSize, Color color) {
  DrawText(TextFormat("FPS: %i", GetFPS()), 10, 10, fontSize, color);
  DrawText(TextFormat("Frame time: %02.02f ms", GetFrameTime()), 60, 10,
           fontSize, color);
}

void switchFullScreen() {
  if (IsKeyPressed(KEY_F11))
    ToggleFullscreen();
}

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
