#pragma once
#include "../../include/raygui.h"
#include "draw.h"
#include "raylib.h"
#include <algorithm>
#include <deque>
#include <vector>
#include <cstdio>

// -------------------------------------------------------------------------
// ENUMS Y CONFIGURACIÓN
// -------------------------------------------------------------------------
enum class ActivationType { ReLU, Tanh, Sigmoid, Linear, Softmax };
enum class CostType { MSE, CrossEntropy, MAE };
enum class OptimizerType { Adam, SGD };

struct ModelConfig {
  std::vector<int> topology;
  ActivationType hiddenActivation;
  ActivationType outputActivation;
  CostType costFunction;
  OptimizerType optimizer;
  float learningRate;
};

// -------------------------------------------------------------------------
// CLASE NETWORK GUI
// -------------------------------------------------------------------------
class NetworkGui {
public:
  int numHiddenLayers = 2;
  int neuronsPerLayer[10] = {20, 20, 20, 20, 20, 20, 20, 20, 20, 20};

  // Dropdowns
  int hiddenActIndex = 0;
  bool hiddenActEdit = false;
  int outputActIndex = 1;
  bool outputActEdit = false;
  int costIndex = 1;
  bool costEdit = false;
  int optimizerIndex = 0;
  bool optimizerEdit = false;

  float learningRate = 0.01f;
  int activeControl = -1;
  bool rebuildRequested = false;
  bool sampleChanged = false;

  // --- GRÁFICA DUAL: TRAIN vs VALIDATION ---
  std::deque<double> trainLossHistory;
  std::deque<double> valLossHistory;
  const size_t maxHistorySize = 200;

  // Método modificado para recibir ambos valores
  void AddLosses(double trainLoss, double valLoss) {
    trainLossHistory.push_back(trainLoss);
    valLossHistory.push_back(valLoss);

    if (trainLossHistory.size() > maxHistorySize) {
      trainLossHistory.pop_front();
      valLossHistory.pop_front();
    }
  }

  // Limpiar ambas gráficas
  void ClearHistory() {
    trainLossHistory.clear();
    valLossHistory.clear();
  }

  void DrawLossGraph(Rectangle bounds) {
    DrawRectangleRec(bounds, Fade(BLACK, 0.8f));
    DrawRectangleLinesEx(bounds, 1.0f, DARKGRAY);

    // Leyenda
    DrawText("Loss: Train (Grn) vs Val (Yel)", (int)bounds.x + 5,
             (int)bounds.y + 5, 10, GRAY);

    if (trainLossHistory.empty())
      return;

    // Scale Plot based on max value of trainLoss/valLoss
    auto minmaxTrain =
        std::minmax_element(trainLossHistory.begin(), trainLossHistory.end());
    auto minmaxVal =
        std::minmax_element(valLossHistory.begin(), valLossHistory.end());

    float maxVal = std::max(*minmaxTrain.second, *minmaxVal.second);
    float minVal = std::min(*minmaxTrain.first, *minmaxVal.first);

    if (maxVal <= minVal)
      maxVal = minVal + 1.0f;

    float stepX = bounds.width / (float)(maxHistorySize - 1);

    // Lambda function to draw a line
    auto DrawHistoryLine = [&](const std::deque<double> &history, Color color) {
      for (size_t i = 0; i < history.size() - 1; i++) {
        float val1 = (float)history[i];
        float val2 = (float)history[i + 1];

        float y1 = bounds.y + bounds.height -
                   ((val1 / maxVal) * (bounds.height - 20)) - 10;
        float y2 = bounds.y + bounds.height -
                   ((val2 / maxVal) * (bounds.height - 20)) - 10;
        float x1 = bounds.x + (i * stepX);
        float x2 = bounds.x + ((i + 1) * stepX);

        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, 2.0f, color);
      }
    };

    // Plot Lines
    DrawHistoryLine(valLossHistory, GOLD);    // Validación (Fondo)
    DrawHistoryLine(trainLossHistory, GREEN); // Train (Frente)

    // Annotate current values
    DrawText(TextFormat("T: %.4f", trainLossHistory.back()),
             (int)bounds.x + 120, (int)bounds.y + 25, 10, GREEN);
    DrawText(TextFormat("V: %.4f", valLossHistory.back()), (int)bounds.x + 170,
             (int)bounds.y + 25, 10, GOLD);
  }

  /*******************************************************************************************************
   *
   * Maind Method to draw
   *
   *********************************************************************************************************/

  void Draw(int screenWidth, int screenHeight, size_t &sampleId,
            size_t totalSamples, DigitViewer &viewer, Vector2 &dataPos,
            float scale) {

    // Set the Box
    Rectangle panelRec = {10, 30, 300, (float)screenHeight - 50};
    GuiGroupBox(panelRec, "Simulation of ANN");

    float startY = 40;
    float spacing = 35;

    // Layers Setup
    GuiLabel((Rectangle){25, startY, 150, 20}, "Hidden Layers:");
    if (GuiSpinner((Rectangle){160, startY, 90, 25}, NULL, &numHiddenLayers, 1,
                   8, activeControl == 99)) {
      activeControl = (activeControl == 99) ? -1 : 99;
    }

    for (int i = 0; i < numHiddenLayers; i++) {
      float yPos = startY + spacing + (i * spacing);
      GuiLabel((Rectangle){25, yPos, 100, 20}, TextFormat("Layer %d:", i + 1));
      if (GuiValueBox((Rectangle){160, yPos, 90, 25}, NULL, &neuronsPerLayer[i],
                      1, 64, activeControl == i)) {
        activeControl = (activeControl == i) ? -1 : i;
      }
    }

    float controlsY = startY + spacing + (numHiddenLayers * spacing) + 10;

    // Hyperparameters
    Rectangle hiddenDropRec = {160, controlsY, 90, 25};
    Rectangle outputDropRec = {160, controlsY + 35, 90, 25};
    Rectangle costDropRec = {160, controlsY + 70, 90, 25};
    Rectangle optimDropRec = {160, controlsY + 105, 90, 25};

    GuiLabel((Rectangle){25, controlsY, 130, 25}, "Hidden Act.:");
    GuiLabel((Rectangle){25, controlsY + 35, 130, 25}, "Output Act.:");
    GuiLabel((Rectangle){25, controlsY + 70, 130, 25}, "Cost Func.:");
    GuiLabel((Rectangle){25, controlsY + 105, 130, 25}, "Optimizer:");

    float lrY = controlsY + 140;
    GuiLabel((Rectangle){25, lrY, 130, 25}, "Learning Rate:");
    GuiSlider((Rectangle){160, lrY, 90, 20}, NULL,
              TextFormat("%.4f", learningRate), &learningRate, 0.0001f, 0.1f);

    float buttonY = lrY + 35;
    if (GuiButton((Rectangle){25, buttonY, 225, 35}, "#103# Compile Model")) {
      rebuildRequested = true;
      activeControl = -1;
      hiddenActEdit = false;
      outputActEdit = false;
      costEdit = false;
      optimizerEdit = false;
    }

    // GRAPH
    float graphY = buttonY + 45;
    Rectangle graphBounds = {25, graphY, 225, 90};
    DrawLossGraph(graphBounds);

    // VIEWER CONTROLS
    float sampleY = graphY + 100;

    // Viewer Rendering
    viewer.draw(dataPos, 0.0f, scale);
    DrawText(TextFormat("Test Sample: %d", sampleId), (int)dataPos.x,
             (int)dataPos.y + (int)(8 * scale) + 10, 20, RAYWHITE);

    // DROPDOWNS
    if (GuiDropdownBox(optimDropRec, "Adam;SGD", &optimizerIndex,
                       optimizerEdit)) {
      optimizerEdit = !optimizerEdit;
      costEdit = false;
      outputActEdit = false;
      hiddenActEdit = false;
    }
    if (GuiDropdownBox(costDropRec, "MSE;CrossEntropy;MAE", &costIndex,
                       costEdit)) {
      costEdit = !costEdit;
      optimizerEdit = false;
      outputActEdit = false;
      hiddenActEdit = false;
    }
    if (GuiDropdownBox(outputDropRec, "Linear;Softmax", &outputActIndex,
                       outputActEdit)) {
      outputActEdit = !outputActEdit;
      optimizerEdit = false;
      costEdit = false;
      hiddenActEdit = false;
    }
    if (GuiDropdownBox(hiddenDropRec, "ReLU;Tanh;Sigmoid", &hiddenActIndex,
                       hiddenActEdit)) {
      hiddenActEdit = !hiddenActEdit;
      optimizerEdit = false;
      outputActEdit = false;
      costEdit = false;
    }
  }

  /*****************************************************************************************************
   *
   * Method to get the Model Configuration to compile the model
   *
   *******************************************************************************************************/
  ModelConfig GetConfig(int inputSize, int outputSize) {

    ModelConfig config;

    config.learningRate = learningRate;
    config.topology.push_back(inputSize);

    for (int i = 0; i < numHiddenLayers; i++)
      config.topology.push_back(neuronsPerLayer[i]);
    config.topology.push_back(outputSize);

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
