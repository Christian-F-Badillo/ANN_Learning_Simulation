#pragma once
#include "../../include/raygui.h"
#include "draw.h"
#include "raylib.h"
#include <algorithm>
#include <deque>
#include <vector>

// -------------------------------------------------------------------------
// ENUMS DE CONFIGURACIÓN
// -------------------------------------------------------------------------
enum class ActivationType { ReLU, Tanh, Sigmoid, Linear, Softmax };
enum class CostType { MSE, CrossEntropy, MAE };
enum class OptimizerType { Adam, SGD }; // Nuevo Enum

// -------------------------------------------------------------------------
// ESTRUCTURA DE CONFIGURACIÓN
// -------------------------------------------------------------------------
struct ModelConfig {
  std::vector<int> topology;
  ActivationType hiddenActivation;
  ActivationType outputActivation;
  CostType costFunction;
  OptimizerType optimizer; // Nuevo campo
  float learningRate;      // Nuevo campo
};

// -------------------------------------------------------------------------
// CLASE NETWORK GUI
// -------------------------------------------------------------------------
class NetworkGui {
public:
  // Estado de Topología
  int numHiddenLayers = 2;
  int neuronsPerLayer[10] = {20, 20, 20, 20, 20, 20, 20, 20, 20, 20};

  // Dropdowns States
  int hiddenActIndex = 0; // 0:ReLU, 1:Tanh, 2:Sigmoid
  bool hiddenActEdit = false;

  int outputActIndex = 1; // 0:Linear, 1:Softmax
  bool outputActEdit = false;

  int costIndex = 1; // 0:MSE, 1:CrossEntropy, 2:MAE
  bool costEdit = false;

  int optimizerIndex = 0; // 0:Adam, 1:SGD (Nuevo)
  bool optimizerEdit = false;

  // Hyperparameters
  float learningRate = 0.01f; // Nuevo

  // Estado General
  int activeControl = -1;
  bool rebuildRequested = false;
  bool sampleChanged = false;

  // Historial de Loss
  std::deque<double> lossHistory;
  const size_t maxHistorySize = 200;

  void AddLoss(float loss) {
    lossHistory.push_back(loss);
    if (lossHistory.size() > maxHistorySize)
      lossHistory.pop_front();
  }

  void DrawLossGraph(Rectangle bounds) {
    DrawRectangleRec(bounds, Fade(BLACK, 0.8f));
    DrawRectangleLinesEx(bounds, 1.0f, DARKGRAY);
    DrawText("Training Loss", (int)bounds.x + 5, (int)bounds.y + 5, 10, GRAY);

    if (lossHistory.empty())
      return;

    float maxVal = *std::max_element(lossHistory.begin(), lossHistory.end());
    float minVal = *std::min_element(lossHistory.begin(), lossHistory.end());
    if (maxVal <= minVal)
      maxVal = minVal + 1.0f;

    float stepX = bounds.width / (float)(maxHistorySize - 1);

    for (size_t i = 0; i < lossHistory.size() - 1; i++) {
      float val1 = lossHistory[i];
      float val2 = lossHistory[i + 1];

      float y1 = bounds.y + bounds.height -
                 ((val1 / maxVal) * (bounds.height - 20)) - 10;
      float y2 = bounds.y + bounds.height -
                 ((val2 / maxVal) * (bounds.height - 20)) - 10;
      float x1 = bounds.x + (i * stepX);
      float x2 = bounds.x + ((i + 1) * stepX);

      Color lineColor = (val2 < val1) ? GREEN : ORANGE;
      DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, 2.0f, lineColor);
    }
    DrawText(TextFormat("%.4f", lossHistory.back()), (int)bounds.x + 5,
             (int)bounds.y + bounds.height - 20, 10, WHITE);
  }

  void Draw(int screenWidth, int screenHeight, size_t &sampleId,
            size_t totalSamples, DigitViewer &viewer, Vector2 &dataPos,
            float scale) {

    Rectangle panelRec = {10, 30, 300, (float)screenHeight - 50};
    GuiGroupBox(panelRec, "ANN Learning Sim");

    float startY = 40;
    float spacing = 30;

    // --- Layers Setup ---
    GuiLabel((Rectangle){25, startY, 150, 20}, "Hidden Layers:");
    if (GuiSpinner((Rectangle){160, startY, 90, 25}, NULL, &numHiddenLayers, 1,
                   8, activeControl == 99)) {
      activeControl = (activeControl == 99) ? -1 : 99;
    }

    for (int i = 0; i < numHiddenLayers; i++) {
      float yPos = startY + spacing + (i * spacing);
      GuiLabel((Rectangle){25, yPos, 100, 20}, TextFormat("Capa %d:", i + 1));
      if (GuiValueBox((Rectangle){160, yPos, 90, 25}, NULL, &neuronsPerLayer[i],
                      1, 128, activeControl == i)) {
        activeControl = (activeControl == i) ? -1 : i;
      }
    }

    float controlsY = startY + spacing + (numHiddenLayers * spacing) + 10;

    // --- Hyperparameters & Functions ---
    // Definir Rectángulos
    Rectangle hiddenDropRec = {160, controlsY, 90, 25};
    Rectangle outputDropRec = {160, controlsY + 35, 90, 25};
    Rectangle costDropRec = {160, controlsY + 70, 90, 25};
    Rectangle optimDropRec = {160, controlsY + 105, 90, 25}; // Nuevo

    GuiLabel((Rectangle){25, controlsY, 130, 25}, "Hidden Act.:");
    GuiLabel((Rectangle){25, controlsY + 35, 130, 25}, "Output Act.:");
    GuiLabel((Rectangle){25, controlsY + 70, 130, 25}, "Cost Func.:");
    GuiLabel((Rectangle){25, controlsY + 105, 130, 25}, "Optimizer:"); // Nuevo

    // Slider Learning Rate
    float lrY = controlsY + 140;
    GuiLabel((Rectangle){25, lrY, 130, 25}, "Learning Rate:");
    GuiSlider((Rectangle){160, lrY, 90, 20}, NULL,
              TextFormat("%.4f", learningRate), &learningRate, 0.0001f, 0.1f);

    // --- BOTONES DE ACCIÓN ---
    float buttonY = lrY + 35;

    if (GuiButton((Rectangle){25, buttonY, 225, 35}, "#103# Compile Model")) {
      rebuildRequested = true;
      activeControl = -1;
      // Cerrar todos los menús
      hiddenActEdit = false;
      outputActEdit = false;
      costEdit = false;
      optimizerEdit = false;
    }

    // --- GRAPH ---
    float graphY = buttonY + 45;
    Rectangle graphBounds = {25, graphY, 225, 90};
    DrawLossGraph(graphBounds);

    viewer.draw(dataPos, 0.0f, scale);

    // --- DIBUJO DE DROPDOWNS (Orden Inverso para Z-Index) ---

    // 4. Optimizer (Nuevo)
    if (GuiDropdownBox(optimDropRec, "Adam;SGD", &optimizerIndex,
                       optimizerEdit)) {
      optimizerEdit = !optimizerEdit;
      costEdit = false;
      outputActEdit = false;
      hiddenActEdit = false;
    }

    // 3. Cost Function
    if (GuiDropdownBox(costDropRec, "MSE;CrossEntropy;MAE", &costIndex,
                       costEdit)) {
      costEdit = !costEdit;
      optimizerEdit = false;
      outputActEdit = false;
      hiddenActEdit = false;
    }

    // 2. Output Activation
    if (GuiDropdownBox(outputDropRec, "Linear;Softmax", &outputActIndex,
                       outputActEdit)) {
      outputActEdit = !outputActEdit;
      optimizerEdit = false;
      costEdit = false;
      hiddenActEdit = false;
    }

    // 1. Hidden Activation
    if (GuiDropdownBox(hiddenDropRec, "ReLU;Tanh;Sigmoid", &hiddenActIndex,
                       hiddenActEdit)) {
      hiddenActEdit = !hiddenActEdit;
      optimizerEdit = false;
      outputActEdit = false;
      costEdit = false;
    }
  }

  ModelConfig GetConfig(int inputSize, int outputSize) {
    ModelConfig config;

    config.learningRate = learningRate;

    // Topology
    config.topology.push_back(inputSize);
    for (int i = 0; i < numHiddenLayers; i++)
      config.topology.push_back(neuronsPerLayer[i]);
    config.topology.push_back(outputSize);

    // Hidden Act
    switch (hiddenActIndex) {
    case 0:
      config.hiddenActivation = ActivationType::ReLU;
      break;
    case 1:
      config.hiddenActivation = ActivationType::Tanh;
      break;
    case 2:
      config.hiddenActivation = ActivationType::Sigmoid;
      break;
    default:
      config.hiddenActivation = ActivationType::ReLU;
      break;
    }

    // Output Act
    switch (outputActIndex) {
    case 0:
      config.outputActivation = ActivationType::Linear;
      break;
    case 1:
      config.outputActivation = ActivationType::Softmax;
      break;
    default:
      config.outputActivation = ActivationType::Softmax;
      break;
    }

    // Cost Func
    switch (costIndex) {
    case 0:
      config.costFunction = CostType::MSE;
      break;
    case 1:
      config.costFunction = CostType::CrossEntropy;
      break;
    case 2:
      config.costFunction = CostType::MAE;
      break;
    default:
      config.costFunction = CostType::CrossEntropy;
      break;
    }

    // Optimizer (Nuevo)
    switch (optimizerIndex) {
    case 0:
      config.optimizer = OptimizerType::Adam;
      break;
    case 1:
      config.optimizer = OptimizerType::SGD;
      break;
    default:
      config.optimizer = OptimizerType::Adam;
      break;
    }

    rebuildRequested = false;
    return config;
  }
};
