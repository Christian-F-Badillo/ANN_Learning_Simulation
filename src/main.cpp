#include "gui/draw.h"
#include "raylib.h"
#include "utils/data_loader.h"
#include <vector>

// Macros
#define WINDOW_HEIGHT 1000
#define WINDOW_WIDTH 1500
#define FPS 60

void switchFullScreen();

int main(int argc, char *argv[]) {

  // Window Config
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT |
                 FLAG_FULLSCREEN_MODE);
  // Create window and Opengl Context
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
             "Simulation of Artificial Neural Networks");

  // Set Frame per Second
  SetTargetFPS(60);

  float radius{50.0f};
  int fontSize{20};
  float scale{10.0f};
  float rotation{0.0f};

  Topology topology{64, 20, 9};

  dataLoader::DataLoader data("../data/optdigits.tra");
  data.loadData();

  int sampleId{100};
  const std::vector<int> &feature = data.getFeatures()[sampleId];

  DigitViewer viewer;
  viewer.setData(feature);

  NetworkLayout layout = calculateNetworkLayout(topology, GetScreenWidth(),
                                                GetScreenHeight(), radius);

  Vector2 dataSamplePosition = {((float)GetScreenWidth()) * 0.05f,
                                (float)GetScreenHeight() / 2};
  // Main app running
  while (!WindowShouldClose()) {

    switchFullScreen();

    if (IsWindowResized()) {
      layout = calculateNetworkLayout(topology, GetScreenWidth(),
                                      GetScreenHeight(), radius);
      Vector2 dataSamplePosition = {((float)GetScreenWidth()) * 0.05f,
                                    (float)GetScreenHeight() / 2};
    }

    BeginDrawing();
    ClearBackground(BLACK);

    drawFPSInfo(fontSize / 2, GREEN);
    drawNetwork(layout);
    drawNetworkConnections(layout);
    viewer.draw(dataSamplePosition, rotation, scale);

    DrawText(TextFormat("Sample: %d", sampleId), dataSamplePosition.x,
             dataSamplePosition.y + (fontSize) + (8 * scale), fontSize, RED);

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

void switchFullScreen() {
  if (IsKeyPressed(KEY_F11))
    ToggleFullscreen();
}
