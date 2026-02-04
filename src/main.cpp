#include <iostream>
#include <vector>
#include <memory>
#include <string>

// -----------------------------------------------------------------------------
// RAYLIB CONF & GUI
// -----------------------------------------------------------------------------
#include "raylib.h"
// clang-format off
#define RAYGUI_IMPLEMENTATION
#include "../include/raygui.h"
#undef RAYGUI_IMPLEMENTATION
#include "../include/style_dark.h"
// clang-format on

// -----------------------------------------------------------------------------
// MÓDULOS DEL PROYECTO
// -----------------------------------------------------------------------------
#include "math/matrix.h"
#include "utils/data_loader.h"
#include "utils/encoding.h"

// Red Neuronal
#include "nn/model.h"
#include "nn/layers.h"
#include "nn/ops.h"
#include "nn/cost_func.h"
#include "nn/optimizer.h"
#include "nn/activation_func.h"

// GUI y Visualización
#include "gui/gui_panel.h"
#include "gui/draw.h"

#define FPS 60

// --- CONSTANTES DE DISEÑO (Adapta el margen izquierdo según el SO) ---
#ifdef __APPLE__
    const float GUI_PANEL_WIDTH = 280.0f; // Un poco más ancho en Retina
#else
    const float GUI_PANEL_WIDTH = 260.0f;
#endif

// --- PROTOTIPOS ---
void CheckWindowResize(NetworkLayout &layout, Vector2 &dataPos, Topology &topo, double radius, double digitScale);
void ToggleAppFullscreen();

int main(int argc, char *argv[]) {
  // -------------------------------------------------------------------------
  // 1. INICIALIZACIÓN DE VENTANA Y ESTILO
  // -------------------------------------------------------------------------
  
  #ifdef __APPLE__
    // macOS: Activar soporte HighDPI para pantallas Retina
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_WINDOW_HIGHDPI);
    InitWindow(1280, 800, "BrainSim (macOS Retina)");
  #else
    // Linux/Windows: Configuración estándar
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(GetScreenWidth(), GetScreenHeight(), "BrainSim (Linux/Win)");
  #endif

  SetTargetFPS(FPS);
  GuiLoadStyleDark();
  GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, 0x838383FF);

  // --- ESCALADO VISUAL DINÁMICO ---
  #ifdef __APPLE__
    float dpiScale = GetWindowScaleDPI().x;
    // Si hay HighDPI, aumentamos el tamaño de los elementos
    double neuronRadius = 24.0 * (dpiScale > 1.0f ? 1.1 : 1.0);
    double digitScale = 9.0 * (dpiScale > 1.0f ? 1.1 : 1.0);
  #else
    double neuronRadius = 25.0;
    double digitScale = 10.0;
  #endif

  // Posición inicial del visualizador: DENTRO del panel izquierdo (abajo)
  // X = 50px (margen izquierdo), Y = Altura - 220px (espacio para controles arriba)
  Vector2 dataSamplePos = { GUI_PANEL_WIDTH /2.0f - (4.0f*(float)digitScale), (float)GetScreenHeight()*0.8f };

  // -------------------------------------------------------------------------
  // 2. CARGA DE DATOS
  // -------------------------------------------------------------------------
  std::cout << "[INFO] Loading Optdigits Dataset..." << std::endl;
  Data::DataLoader data("../data/optdigits.tra");
  try {
    data.loadData();
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return -1;
  }

  const auto &rawFeatures = data.getFeatures();
  const auto &rawLabels = data.getLabels();
  size_t totalSamples = rawFeatures.shape()[0];
  size_t inputSize = rawFeatures.shape()[1]; // 64
  size_t outputSize = 10;                    // 0-9

  // Pre- procesar datos a double para entrenamiento (evita conversiones en cada frame)
  std::vector<double> allFeaturesDouble(rawFeatures.data().begin(), rawFeatures.data().end());
  Math::Matrix<double> X_train_all(std::move(allFeaturesDouble), rawFeatures.shape());
  Math::Matrix<double> y_train_all = Data::Encoder::toOneHot<double>(rawLabels, (int)outputSize);

  // -------------------------------------------------------------------------
  // 3. ESTADO DE LA SIMULACIÓN
  // -------------------------------------------------------------------------
  NetworkGui gui;
  size_t currentSampleId = 0;
  
  // Visor de imagen de entrada
  DigitViewer viewer;
  viewer.setData(rawFeatures.atRow(currentSampleId).data());

  // Configuración inicial de Topología
  Topology topology = {(int)inputSize, 20, 20, (int)outputSize}; 
  
  // Calculamos layout inicial pasando el ancho del panel para el offset
  NetworkLayout layout = calculateNetworkLayout(topology, GetScreenWidth(), GetScreenHeight(), (float)neuronRadius, GUI_PANEL_WIDTH);

  // -------------------------------------------------------------------------
  // 4. MODELO DE RED NEURONAL
  // -------------------------------------------------------------------------
  NN::Model<double> model;
  std::shared_ptr<NN::Optimizer::Optimizer<double>> optimizer = nullptr;

  // Forzar construcción inicial
  gui.rebuildRequested = true;
  
  int predictedLabel = -1;
  int targetLabel = -1;

  // -------------------------------------------------------------------------
  // 5. BUCLE PRINCIPAL
  // -------------------------------------------------------------------------
  while (!WindowShouldClose()) {

    ToggleAppFullscreen();
    CheckWindowResize(layout, dataSamplePos, topology, neuronRadius, digitScale);

    // --- RECONSTRUCCIÓN DEL MODELO (Si cambia la GUI) ---
    if (gui.rebuildRequested) {
      ModelConfig cfg = gui.GetConfig((int)inputSize, (int)outputSize);
      
      // Actualizar visuales
      topology = cfg.topology;
      layout = calculateNetworkLayout(topology, GetScreenWidth(), GetScreenHeight(), (float)neuronRadius, GUI_PANEL_WIDTH);
      
      // Construir Red Secuencial
      auto sequential = std::make_shared<NN::Layer::Sequential<double>>();
      
      // Capas Ocultas
      for (size_t i = 1; i < cfg.topology.size() - 1; ++i) {
        std::shared_ptr<NN::Ops::Operation<double>> act;
        switch (cfg.hiddenActivation) {
            case ActivationType::ReLU:    act = std::make_shared<NN::ActFunc::ReLU<double>>(); break;
            case ActivationType::Tanh:    act = std::make_shared<NN::ActFunc::Tanh<double>>(); break;
            case ActivationType::Sigmoid: act = std::make_shared<NN::ActFunc::Sigmoid<double>>(); break;
            default:                      act = std::make_shared<NN::ActFunc::ReLU<double>>(); break;
        }
        sequential->add(std::make_shared<NN::Layer::Dense<double>>(cfg.topology[i], act));
      }
      
      // Capa de Salida
      std::shared_ptr<NN::Ops::Operation<double>> outAct;
      if (cfg.outputActivation == ActivationType::Softmax) 
          outAct = std::make_shared<NN::ActFunc::Softmax<double>>();
      else 
          outAct = std::make_shared<NN::ActFunc::Linear<double>>();
          
      sequential->add(std::make_shared<NN::Layer::Dense<double>>((int)outputSize, outAct));

      // Función de Costo
      std::shared_ptr<NN::CostFunc::Loss<double>> loss;
      switch (cfg.costFunction) {
        case CostType::MSE:          loss = std::make_shared<NN::CostFunc::MeanSquareError<double>>(); break;
        case CostType::MAE:          loss = std::make_shared<NN::CostFunc::MeanAbsoluteError<double>>(); break;
        case CostType::CrossEntropy: loss = std::make_shared<NN::CostFunc::CategoricalCrossEntropy<double>>(); break;
      }

      // Optimizador
      if (cfg.optimizer == OptimizerType::Adam) 
          optimizer = std::make_shared<NN::Optimizer::Adam<double>>(cfg.learningRate);
      else 
          optimizer = std::make_shared<NN::Optimizer::SGD<double>>(cfg.learningRate);

      model.set_layers(sequential);
      model.compile(loss, optimizer);
      gui.lossHistory.clear();
    }

    // --- INPUTS: ENTRENAMIENTO Y NAVEGACIÓN ---
    
    // 1. ESPACIO: Entrenar 1 Paso con TODO el dataset (Full Batch)
    if (IsKeyPressed(KEY_SPACE)) {
        double lossVal = model.train_step(X_train_all, y_train_all);
        gui.AddLoss((float)lossVal);
    }

    // 2. FLECHAS: Cambiar muestra visualizada (Inferencia)
    if (IsKeyPressed(KEY_RIGHT)) { 
        currentSampleId = (currentSampleId + 1) % totalSamples; 
        gui.sampleChanged = true; 
    }
    if (IsKeyPressed(KEY_LEFT)) { 
        currentSampleId = (currentSampleId == 0) ? totalSamples - 1 : currentSampleId - 1; 
        gui.sampleChanged = true; 
    }

    // Actualizar textura si cambió la muestra
    if (gui.sampleChanged) { 
        viewer.setData(rawFeatures.atRow(currentSampleId).data()); 
        gui.sampleChanged = false; 
    }

    // --- INFERENCIA CONTINUA (Sobre la muestra actual en pantalla) ---
    std::vector<double> curRow = X_train_all.atRow(currentSampleId).data();
    Math::Matrix<double> x_in(curRow, {1, (int)inputSize});
    
    // Predecir y obtener etiqueta
    auto predictionMat = model.predict(x_in);
    predictedLabel = Data::Encoder::argMax(predictionMat.data());
    targetLabel = rawLabels.atRow(currentSampleId).data()[0];

    // --- RENDERIZADO ---
    BeginDrawing();
    ClearBackground(BLACK);

    drawFPSInfo(10, GREEN);

    // 1. Dibujar Red Neuronal (Fondo)
    drawNetwork(layout);
    drawNetworkConnections(layout, model.get_parameters());

    // 2. Dibujar Viewer (Imagen del dígito)
    viewer.draw(dataSamplePos, 0.0f, (float)digitScale);
    
    // 3. Textos Informativos (Debajo del viewer)
    int textY = (int)dataSamplePos.y + (8 * (int)digitScale) + 10;
    
    // Resultado Predicción
    Color resultColor = (predictedLabel == targetLabel) ? GREEN : RED;
    DrawText(TextFormat("Pred: %d (Real: %d)", predictedLabel, targetLabel), 
             (int)dataSamplePos.x/2.0f, textY + 10, 20, resultColor); // Bajamos un poco para no tapar "Target" de GUI

    // Instrucciones (Alineadas dentro del panel)
    #ifdef __APPLE__
        DrawText("ESPACIO: Entrenar", (int)200, 10, 10, GRAY);
        DrawText("FLECHAS: Navegar", (int)325, 10, 10, GRAY);
    #else
        DrawText("SPACE: Train Batch", (int)dataSamplePos.x, textY + 45, 10, GRAY);
        DrawText("ARROWS: Navigate", (int)dataSamplePos.x, textY + 60, 10, GRAY);
    #endif

    // 4. GUI Panel (Siempre encima)
    gui.Draw(GetScreenWidth(), GetScreenHeight(), currentSampleId, totalSamples, viewer, dataSamplePos, (float)digitScale);

    EndDrawing();
  }

  CloseWindow();
  return 0;
}

// -----------------------------------------------------------------------------
// IMPLEMENTACIÓN HELPERS
// -----------------------------------------------------------------------------

void ToggleAppFullscreen() { 
    if (IsKeyPressed(KEY_T)) ToggleFullscreen(); 
}

void CheckWindowResize(NetworkLayout &layout, Vector2 &dataPos, Topology &topo, double radius, double digitScale) {
  if (IsWindowResized()) {
    // Recalcular layout pasando el ancho del panel para evitar solapamiento
    layout = calculateNetworkLayout(topo, GetScreenWidth(), GetScreenHeight(), (float)radius, GUI_PANEL_WIDTH);
    
    // Reposicionar el visor de datos DENTRO del panel
    dataPos = { GUI_PANEL_WIDTH/2.0f - (4.0f*(float) digitScale), (float)GetScreenHeight()*0.55f};
  }
}