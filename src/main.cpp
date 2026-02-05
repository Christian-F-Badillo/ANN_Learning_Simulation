#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "../include/raygui.h"
#undef RAYGUI_IMPLEMENTATION
#include "../include/style_dark.h"

#include "math/matrix.h"
#include "utils/data_loader.h"
#include "utils/encoding.h"
#include "utils/split_shuffle.h"

#include "nn/model.h"
#include "nn/layers.h"
#include "nn/ops.h"
#include "nn/cost_func.h"
#include "nn/optimizer.h"
#include "nn/activation_func.h"

#include "gui/gui_panel.h"
#include "gui/draw.h"

// Macros
#define FPS 60

#ifdef __APPLE__
const float GUI_PANEL_WIDTH = 280.0f;
#else
const float GUI_PANEL_WIDTH = 280.0f;
#endif

/*********************************************************************************************************
 *
 * Auxiliar Function Declarations
 *
 *********************************************************************************************************/

void CheckWindowResize(NetworkLayout &layout, Vector2 &dataPos, Topology &topo,
                       double radius);
void ToggleAppFullscreen();
void RebuildNetworkModel(NetworkGui &gui, NN::Model<double> &model,
                         Topology &topology, NetworkLayout &layout,
                         int inputSize, int outputSize, int screenW,
                         int screenH, double radius, float panelW);

/*********************************************************************************************************
 *
 * MAIN FUNCTION
 *
 *********************************************************************************************************/

int main(int argc, char *argv[]) {

  // Init Window
#ifdef __APPLE__
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT |
                 FLAG_WINDOW_HIGHDPI);
  InitWindow(1280, 800, "Simulation of Artificial Neural Network");
#else
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
  InitWindow(GetScreenWidth(), GetScreenHeight(),
             "Simulation of Artificial Neural Network");
#endif

  // Set FPS reder target
  SetTargetFPS(FPS);

  // Set the GUI Style

  GuiLoadStyleDark();
  GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, 0x838383FF);

  // -------------------------------------------------------------------------
  // Set Up of Neurons and Viewer Scale
  // -------------------------------------------------------------------------

#ifdef __APPLE__
  float dpiScale = GetWindowScaleDPI().x;
  double neuronRadius = 15.0 * (dpiScale > 1.0f ? 1.1 : 1.0);
  double digitScale = 9.0 * (dpiScale > 1.0f ? 1.1 : 1.0);
#else
  double neuronRadius = 15.0;
  double digitScale = 9.0;
#endif

  Vector2 dataSamplePos = {50.0f, (float)GetScreenHeight() - 220.0f};

  // -------------------------------------------------------------------------
  // Training Data
  // -------------------------------------------------------------------------

  // Load Training Data
  std::cout << "[INFO] Loading Training Source (optdigits.tra)..." << std::endl;
  Data::DataLoader trainSource("../data/optdigits.tra");
  try {
    trainSource.loadData();
  } catch (...) {
    return -1;
  }

  // Get Features
  const auto &srcFeat = trainSource.getFeatures();
  // Get Labels
  const auto &srcLabels = trainSource.getLabels();
  // Number of Features (inputSize)
  size_t inputSize = srcFeat.shape()[1];
  // Number of Classes
  size_t outputSize = 10;

  // Create Data as Matrix
  std::vector<double> srcFeatDbl(srcFeat.data().begin(), srcFeat.data().end());
  Math::Matrix<double> X_source(std::move(srcFeatDbl), srcFeat.shape());

  // One-Hot Enconding of Labels
  Math::Matrix<double> Y_source =
      Data::Encoder::toOneHot<double>(srcLabels, (int)outputSize);

  // Split Data in Training and Validation
  std::cout << "[INFO] Split Data..." << std::endl;
  auto dataset = Utils::SplitShuffle::split(X_source, Y_source, 0.8f, 42);

  // -------------------------------------------------------------------------
  // Test Data
  // -------------------------------------------------------------------------

  // Load Test Data
  std::cout << "[INFO] Loading Test Data (optdigits.tes)..." << std::endl;
  Data::DataLoader viewerSource("../data/optdigits.tes");
  try {
    viewerSource.loadData();
  } catch (...) {
    return -1;
  }

  // Get Features and Labels
  const auto &viewerFeatures = viewerSource.getFeatures();
  const auto &viewerLabels = viewerSource.getLabels();
  size_t totalViewerSamples = viewerFeatures.shape()[0];

  // Create Features and labels as Matrix
  std::vector<double> viewerFeatDbl(viewerFeatures.data().begin(),
                                    viewerFeatures.data().end());
  Math::Matrix<double> X_viewer_all(std::move(viewerFeatDbl),
                                    viewerFeatures.shape());

  // -------------------------------------------------------------------------
  // GUI Initial Set Up
  // -------------------------------------------------------------------------
  NetworkGui gui;
  size_t currentSampleId = 0;
  DigitViewer viewer;
  viewer.setData(viewerFeatures.atRow(currentSampleId).data());

  // Initial Network Layout
  Topology topology = {(int)inputSize, 20, 10, (int)outputSize};
  NetworkLayout layout =
      calculateNetworkLayout(topology, GetScreenWidth(), GetScreenHeight(),
                             (float)neuronRadius, GUI_PANEL_WIDTH);

  // Default Model
  NN::Model<double> model;
  std::shared_ptr<NN::Optimizer::Optimizer<double>> optimizer = nullptr;

  std::shared_ptr<NN::CostFunc::Loss<double>> currentLossFunc = nullptr;

  gui.rebuildRequested = true;
  int predictedLabel = -1;
  int targetLabel = -1;

  // -------------------------------------------------------------------------
  // APP LOOP
  // -------------------------------------------------------------------------
  while (!WindowShouldClose()) {
    ToggleAppFullscreen();
    CheckWindowResize(layout, dataSamplePos, topology, neuronRadius);

    // Rebuild the Model
    if (gui.rebuildRequested) {
      RebuildNetworkModel(gui, model, topology, layout, (int)inputSize,
                          (int)outputSize, GetScreenWidth(), GetScreenHeight(),
                          neuronRadius, GUI_PANEL_WIDTH);

      ModelConfig cfg = gui.GetConfig((int)inputSize, (int)outputSize);
      if (cfg.costFunction == CostType::MSE)
        currentLossFunc =
            std::make_shared<NN::CostFunc::MeanSquareError<double>>();
      else if (cfg.costFunction == CostType::MAE)
        currentLossFunc =
            std::make_shared<NN::CostFunc::MeanAbsoluteError<double>>();
      else
        currentLossFunc =
            std::make_shared<NN::CostFunc::CategoricalCrossEntropy<double>>();

      // Initial inference
      std::vector<double> curRow = X_viewer_all.atRow(currentSampleId).data();
      Math::Matrix<double> x_in(curRow, {1, (int)inputSize});
      predictedLabel = Data::Encoder::argMax(model.predict(x_in).data());
      targetLabel = viewerLabels.atRow(currentSampleId).data()[0];
    }

    // Train the Model
    if (IsKeyPressed(KEY_SPACE)) {
      // Forward and Backward pass and get the Training Loss
      double trainLoss = model.train_step(dataset.X_train, dataset.Y_train);

      // Forward on Validation Data and get the Validation Loss
      auto valPreds = model.predict(dataset.X_val);
      double valLoss = currentLossFunc->forward(valPreds, dataset.Y_val);

      // Update the inference
      std::vector<double> curRow = X_viewer_all.atRow(currentSampleId).data();
      Math::Matrix<double> x_in(curRow, {1, (int)inputSize});
      predictedLabel = Data::Encoder::argMax(model.predict(x_in).data());
      targetLabel = viewerLabels.atRow(currentSampleId).data()[0];

      // Update the Loss Plot
      gui.AddLosses(trainLoss, valLoss);
    }

    // Select the Test Sample to Predict using Left and Right Arrows
    if (IsKeyPressed(KEY_RIGHT)) {
      currentSampleId = (currentSampleId + 1) % totalViewerSamples;
      gui.sampleChanged = true;
    }
    if (IsKeyPressed(KEY_LEFT)) {
      currentSampleId =
          (currentSampleId == 0) ? totalViewerSamples - 1 : currentSampleId - 1;
      gui.sampleChanged = true;
    }

    // Only Predict if Test Sample Change
    if (gui.sampleChanged) {
      viewer.setData(viewerFeatures.atRow(currentSampleId).data());
      // Make the inference
      std::vector<double> curRow = X_viewer_all.atRow(currentSampleId).data();
      Math::Matrix<double> x_in(curRow, {1, (int)inputSize});
      predictedLabel = Data::Encoder::argMax(model.predict(x_in).data());
      targetLabel = viewerLabels.atRow(currentSampleId).data()[0];
      gui.sampleChanged = false;
    }

    // --- DIBUJO ---
    BeginDrawing();
    ClearBackground(BLACK);
    drawFPSInfo(10, GREEN);

    drawNetwork(layout);
    drawNetworkConnections(layout, model.get_parameters());
    viewer.draw(dataSamplePos, 0.0f, (float)digitScale);

    int textY = (int)dataSamplePos.y + (8 * (int)digitScale) + 10;
    Color resultColor = (predictedLabel == targetLabel) ? GREEN : RED;
    DrawText(TextFormat("Pred: %d (Real: %d)", predictedLabel, targetLabel),
             (int)dataSamplePos.x, textY + 20, 20, resultColor);

#ifdef __APPLE__
    DrawText("SPACE: Do Epoch Training | LEFT/RIGHT ARROW: Predict Test Sample",
             (int)200, 10, 20, GRAY);
#else
    DrawText("SPACE: Do Epoch Training | LEFT/RIGHT ARROW: Predict Test Sample",
             (int)350, 10, 20, GRAY);
#endif

    gui.Draw(GetScreenWidth(), GetScreenHeight(), currentSampleId,
             totalViewerSamples, viewer, dataSamplePos, (float)digitScale);
    EndDrawing();
  }
  CloseWindow();
  return 0;
}

/********************************************************************************************************************
 *
 * Auxiliar Function Implementation
 *
 **********************************************************************************************************************/

void RebuildNetworkModel(NetworkGui &gui, NN::Model<double> &model,
                         Topology &topology, NetworkLayout &layout,
                         int inputSize, int outputSize, int screenW,
                         int screenH, double radius, float panelW) {

  std::cout << "[INFO] Building new model..." << std::endl;

  // Get new model
  ModelConfig cfg = gui.GetConfig(inputSize, outputSize);

  // Update Draws
  topology = cfg.topology;
  layout =
      calculateNetworkLayout(topology, screenW, screenH, (float)radius, panelW);

  // Init Sequential Layer
  auto sequential = std::make_shared<NN::Layer::Sequential<double>>();

  // Hidden Layers
  for (size_t i = 1; i < cfg.topology.size() - 1; ++i) {
    std::shared_ptr<NN::Ops::Operation<double>> act;
    switch (cfg.hiddenActivation) {
    case ActivationType::ReLU:
      act = std::make_shared<NN::ActFunc::ReLU<double>>();
      break;
    case ActivationType::Tanh:
      act = std::make_shared<NN::ActFunc::Tanh<double>>();
      break;
    case ActivationType::Sigmoid:
      act = std::make_shared<NN::ActFunc::Sigmoid<double>>();
      break;
    default:
      act = std::make_shared<NN::ActFunc::ReLU<double>>();
      break;
    }
    sequential->add(
        std::make_shared<NN::Layer::Dense<double>>(cfg.topology[i], act));
  }

  // Output Layer
  std::shared_ptr<NN::Ops::Operation<double>> outAct;
  if (cfg.outputActivation == ActivationType::Softmax)
    outAct = std::make_shared<NN::ActFunc::Softmax<double>>();
  else
    outAct = std::make_shared<NN::ActFunc::Linear<double>>();
  sequential->add(
      std::make_shared<NN::Layer::Dense<double>>(outputSize, outAct));

  // Conf Loss
  std::shared_ptr<NN::CostFunc::Loss<double>> lossFunc;
  switch (cfg.costFunction) {
  case CostType::MSE:
    lossFunc = std::make_shared<NN::CostFunc::MeanSquareError<double>>();
    break;
  case CostType::MAE:
    lossFunc = std::make_shared<NN::CostFunc::MeanAbsoluteError<double>>();
    break;
  case CostType::CrossEntropy:
    lossFunc =
        std::make_shared<NN::CostFunc::CategoricalCrossEntropy<double>>();
    break;
  }

  // Conf Optimizer
  std::shared_ptr<NN::Optimizer::Optimizer<double>> optimizer;
  if (cfg.optimizer == OptimizerType::Adam)
    optimizer = std::make_shared<NN::Optimizer::Adam<double>>(cfg.learningRate);
  else
    optimizer = std::make_shared<NN::Optimizer::SGD<double>>(cfg.learningRate);

  // Compile Model
  model.set_layers(sequential);
  model.compile(lossFunc, optimizer);

  // Clear Loss Plot and Loss data
  gui.ClearHistory();
}

void ToggleAppFullscreen() {
  if (IsKeyPressed(KEY_T))
    ToggleFullscreen();
}

void CheckWindowResize(NetworkLayout &layout, Vector2 &dataPos, Topology &topo,
                       double radius) {
  if (IsWindowResized()) {
    layout = calculateNetworkLayout(topo, GetScreenWidth(), GetScreenHeight(),
                                    (float)radius, GUI_PANEL_WIDTH);
    dataPos = {50.0f, (float)GetScreenHeight() - 220.0f};
  }
}
