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
// My Modules
// -----------------------------------------------------------------------------
// Matrix Support
#include "math/matrix.h"
// Data utils
#include "utils/data_loader.h"
// Encoding Labels
#include "utils/encoding.h"

// Neural Network
#include "nn/model.h"
#include "nn/layers.h"
#include "nn/ops.h"
#include "nn/cost_func.h"
#include "nn/optimizer.h"
#include "nn/activation_func.h"

// GUI
#include "gui/gui_panel.h"
#include "gui/draw.h"

// Macros
#define FPS 60

// --- PROTOTIPOS ---
void CheckWindowResize(NetworkLayout &layout, Vector2 &dataPos, Topology &topo,
                       double radius);
void ToggleAppFullscreen();

int main(int argc, char *argv[]) {
  // -------------------------------------------------------------------------
  // 1. INICIALIZACIÓN DE VENTANA Y ESTILO
  // -------------------------------------------------------------------------
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT |
                 FLAG_FULLSCREEN_MODE);
  InitWindow(GetScreenWidth(), GetScreenHeight(), "Neural Network Laboratory");
  SetTargetFPS(FPS);

  // Cargar estilo Dark
  GuiLoadStyleDark();
  GuiSetStyle(DEFAULT, TEXT_COLOR_NORMAL, 0x838383FF);

  // Constantes de Dibujo
  double neuronRadius = 25.0f; // Radio base
  double scale = 10.0f;        // Escala del visualizador de dígitos

  // Posición dinámica del visualizador de datos
  Vector2 dataSamplePos = {((float)GetScreenWidth()) * 0.045f,
                           (float)GetScreenHeight() / 2.0f +
                               ((float)GetScreenHeight() / 6.5f)};

  // -------------------------------------------------------------------------
  // 2. CARGA DE DATOS (DATASET)
  // -------------------------------------------------------------------------
  std::cout << "[INFO] Loading Dataset..." << std::endl;
  Data::DataLoader data("../data/optdigits.tra");
  try {
    data.loadData();
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] " << e.what() << std::endl;
    return -1;
  }

  // Referencias a los datos crudos
  const Math::Matrix<int> &rawFeatures = data.getFeatures();
  const Math::Matrix<int> &rawLabels = data.getLabels();
  size_t totalSamples = rawFeatures.shape()[0];
  size_t inputSize = rawFeatures.shape()[1]; // 64
  size_t outputSize = 10;                    // 0-9

  // --- PREPROCESAMIENTO DATOS COMPLETOS (ALL DATA) ---
  std::vector<int> allRawData = rawFeatures.data();
  std::vector<double> allFeaturesDouble(allRawData.begin(), allRawData.end());

  // Matriz con todo el dataset convertido a double
  Math::Matrix<double> X_train_all(std::move(allFeaturesDouble),
                                   rawFeatures.shape());

  // Matriz con todos los targets en One-Hot (double)
  Math::Matrix<double> y_train_all =
      Data::Encoder::toOneHot<double>(rawLabels, (int)outputSize);

  // -------------------------------------------------------------------------
  // 3. ESTADO DE LA SIMULACIÓN
  // -------------------------------------------------------------------------
  NetworkGui gui;
  size_t currentSampleId = 0;

  // Visor de dígitos (Input Image)
  DigitViewer viewer;
  viewer.setData(rawFeatures.atRow(currentSampleId).data());

  // Topología Inicial y Layout
  Topology topology = {(int)inputSize, 20, 20, (int)outputSize}; // Default init
  NetworkLayout layout = calculateNetworkLayout(
      topology, GetScreenWidth(), GetScreenHeight(), neuronRadius);

  // -------------------------------------------------------------------------
  // 4. MODELO DE RED NEURONAL
  // -------------------------------------------------------------------------
  NN::Model<double> model;

  // Optimizador por defecto
  std::shared_ptr<NN::Optimizer::Optimizer<double>> optimizer = nullptr;

  // Forzamos una primera construcción del modelo
  gui.rebuildRequested = true;

  // Variables para inferencia
  int predictedLabel = -1;
  int targetLabel = -1;

  // -------------------------------------------------------------------------
  // 5. BUCLE PRINCIPAL
  // -------------------------------------------------------------------------
  while (!WindowShouldClose()) {

    // --- INPUT & WINDOW HANDLING ---
    ToggleAppFullscreen();
    CheckWindowResize(layout, dataSamplePos, topology, neuronRadius);

    // --- LÓGICA DE RECONSTRUCCIÓN ---
    if (gui.rebuildRequested) {
      std::cout << "[INFO] Rebuilding Model..." << std::endl;

      ModelConfig cfg = gui.GetConfig((int)inputSize, (int)outputSize);
      topology = cfg.topology; // Actualizar topología visual

      // Recalcular layout visual
      layout = calculateNetworkLayout(topology, GetScreenWidth(),
                                      GetScreenHeight(), neuronRadius);

      // 1. Crear Estructura Secuencial
      auto sequential = std::make_shared<NN::Layer::Sequential<double>>();

      // 2. Capas Ocultas
      for (size_t i = 1; i < cfg.topology.size() - 1; ++i) {
        std::shared_ptr<NN::Ops::Operation<double>> actFunc;

        switch (cfg.hiddenActivation) {
        case ActivationType::ReLU:
          actFunc = std::make_shared<NN::ActFunc::ReLU<double>>();
          break;
        case ActivationType::Tanh:
          actFunc = std::make_shared<NN::ActFunc::Tanh<double>>();
          break;
        case ActivationType::Sigmoid:
          actFunc = std::make_shared<NN::ActFunc::Sigmoid<double>>();
          break;
        default:
          actFunc = std::make_shared<NN::ActFunc::ReLU<double>>();
          break;
        }
        sequential->add(std::make_shared<NN::Layer::Dense<double>>(
            cfg.topology[i], actFunc));
      }

      // 3. Capa de Salida
      std::shared_ptr<NN::Ops::Operation<double>> outFunc;
      switch (cfg.outputActivation) {
      case ActivationType::Linear:
        outFunc = std::make_shared<NN::ActFunc::Linear<double>>();
        break;
      case ActivationType::Softmax:
        outFunc = std::make_shared<NN::ActFunc::Softmax<double>>();
        break;
      default:
        outFunc = std::make_shared<NN::ActFunc::Softmax<double>>();
        break;
      }
      sequential->add(
          std::make_shared<NN::Layer::Dense<double>>((int)outputSize, outFunc));

      // 4. Función de Costo
      std::shared_ptr<NN::CostFunc::Loss<double>> lossFunc;
      switch (cfg.costFunction) {
      case CostType::MSE:
        lossFunc = std::make_shared<NN::CostFunc::MeanSquareError<double>>();
        break;
      case CostType::CrossEntropy:
        lossFunc =
            std::make_shared<NN::CostFunc::CategoricalCrossEntropy<double>>();
        break;
      case CostType::MAE:
        lossFunc = std::make_shared<NN::CostFunc::MeanAbsoluteError<double>>();
        break;
      }

      // 5. Compilar Modelo (CON EL OPTIMIZADOR SELECCIONADO)
      switch (cfg.optimizer) {
      case OptimizerType::Adam:
        optimizer =
            std::make_shared<NN::Optimizer::Adam<double>>(cfg.learningRate);
        break;
      case OptimizerType::SGD:
        optimizer =
            std::make_shared<NN::Optimizer::SGD<double>>(cfg.learningRate);
        break;
      }

      model.set_layers(sequential);
      model.compile(lossFunc, optimizer);

      // Limpiar historial de la gráfica
      gui.lossHistory.clear();
    }

    // --- CONTROLES DE USUARIO ---

    // 1. Navegación de Muestras (FLECHAS IZQ/DER)
    if (IsKeyPressed(KEY_RIGHT)) {
      currentSampleId++;
      if (currentSampleId >= totalSamples)
        currentSampleId = 0;
      gui.sampleChanged = true;
    } else if (IsKeyPressed(KEY_LEFT)) {
      if (currentSampleId == 0)
        currentSampleId = totalSamples - 1;
      else
        currentSampleId--;
      gui.sampleChanged = true;
    }

    // 2. Entrenamiento (ESPACIO) - Full Batch
    if (IsKeyPressed(KEY_SPACE)) {
      // Usamos X_train_all y y_train_all que contienen TODO el dataset
      double loss = model.train_step(X_train_all, y_train_all);
      gui.AddLoss(loss);
    }

    // --- ACTUALIZACIÓN DE DATOS (Muestra Actual para Visualización) ---
    // Si cambió el ID, actualizamos el viewer
    if (gui.sampleChanged) {
      viewer.setData(rawFeatures.atRow(currentSampleId).data());
      gui.sampleChanged = false;
    }

    // --- PREPARACIÓN PARA INFERENCIA (Sample Actual) ---
    // Aunque entrenemos con todos, queremos ver qué predice la red para el
    // número en pantalla

    // Extraemos la fila actual de la matriz gigante de doubles que ya creamos
    std::vector<double> currentFeatureRow =
        X_train_all.atRow(currentSampleId).data();
    Math::Matrix<double> x_input_sample(currentFeatureRow,
                                        std::vector<int>{1, (int)inputSize});

    // Inferencia Continua
    Math::Matrix<double> prediction = model.predict(x_input_sample);
    predictedLabel = Data::Encoder::argMax(prediction.data());

    // Etiqueta Real
    targetLabel = rawLabels.atRow(currentSampleId).data()[0];

    // --- RENDERIZADO ---
    BeginDrawing();
    ClearBackground(BLACK);

    // A. Información de Debug
    drawFPSInfo(10, GREEN);

    // B. Red Neuronal (Fondo)
    drawNetwork(layout);
    drawNetworkConnections(layout, model.get_parameters());

    // C. Visualizador de Datos (Input)
    viewer.draw(dataSamplePos, 0.0f, scale);

    // D. Información de Texto (Predicción vs Real)
    double textX = dataSamplePos.x / 1.5f;
    double textY = dataSamplePos.y + (8 * scale) + 40;

    Color predColor = (predictedLabel == targetLabel) ? GREEN : RED;
    DrawText(TextFormat("Prediction: %d", predictedLabel), textX, textY + 10,
             20, predColor);

    // Instrucciones Actualizadas
    DrawText("ESPACIO: Entrenar (Dataset Completo)", textX, textY + 60, 10,
             GRAY);
    DrawText("FLECHAS: Navegar Muestras", textX, textY + 75, 10, GRAY);

    // E. GUI Panel (Siempre al final para Z-Index superior)
    gui.Draw(GetScreenWidth(), GetScreenHeight(), currentSampleId, totalSamples,
             viewer, dataSamplePos, scale);

    EndDrawing();
  }

  // Limpieza
  CloseWindow();
  return 0;
}

// -----------------------------------------------------------------------------
// IMPLEMENTACIÓN DE FUNCIONES AUXILIARES
// -----------------------------------------------------------------------------

void ToggleAppFullscreen() {
  if (IsKeyPressed(KEY_F11)) {
    ToggleFullscreen();
  }
}

void CheckWindowResize(NetworkLayout &layout, Vector2 &dataPos, Topology &topo,
                       double radius) {
  if (IsWindowResized()) {
    // Recalcular layout de la red
    layout = calculateNetworkLayout(topo, GetScreenWidth(), GetScreenHeight(),
                                    radius);

    // Recalcular posición del viewer
    dataPos = {
        ((float)GetScreenWidth()) * 0.045f,
        (float)GetScreenHeight() / 2.0f +
            ((float)GetScreenHeight() / 6.5f) // Centrado vertical aprox
    };
  }
}
