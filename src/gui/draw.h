#pragma once
#include "raylib.h"
#include <vector>

// Custom Types
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

// Functions Declarations
void drawFPSInfo(int fontSize, Color color);
void drawFPSInfo(int fontSize, Color color);
void switchFullScreen();
void drawNetwork(const NetworkLayout &layout);
NetworkLayout calculateNetworkLayout(const Topology &topology, int screenWidth,
                                     int screenHeight, float targetRadius);
NeuronTheme getLayerColors(Layer type);
void drawNetworkConnections(const NetworkLayout &layout);
void drawDataSample(const std::vector<int> &dataSample, Vector2 position,
                    float rotation, float scale);

class DigitViewer {
public:
  DigitViewer();
  ~DigitViewer();
  void setData(const std::vector<int> &dataSample);
  void draw(Vector2 position, float rotation, float scale);

private:
  Texture2D texture;
};
