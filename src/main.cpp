#include "../src/networkdraw.h"
#include "raylib.h"

// Macros
#define WINDOW_HEIGHT 1000
#define WINDOW_WIDTH 1500
#define FPS 60

int main(int argc, char *argv[]) {

  // Window Config
  SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
  // Create window and Opengl Context
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
             "Simulation of Artificial Neural Networks");

  // Set Frame per Second

  float radius{50.0f};
  Topology topology{64, 20, 9};
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
    drawNetworkConnections(layout);

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
