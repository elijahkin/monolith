// Usage: animate [output_directory]
// Example: ./animate /tmp/mandelbrot_frames

#include <cstdlib>
#include <filesystem>
#include <string>

#include "mandelbrot.hpp"

int main(int argc, char *argv[]) {
  const char *workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  const std::filesystem::path output_path =
      argc > 1    ? std::filesystem::path(argv[1])
      : workspace ? std::filesystem::path(workspace)
                  : std::filesystem::current_path();
  const std::string path =
      (workspace && output_path.is_relative())
          ? (std::filesystem::path(workspace) / output_path).string()
          : output_path.string();
  std::filesystem::create_directories(path);

  auto plotter = mandelbrot::Plotter(1920, 1080);

  // Rendering each frame of an animation
  double apothem = 3;
  for (int i = 0; i < 16080; ++i) {
    plotter.Plot(-0.77568377, 0.13646737, apothem, 800, std::to_string(i),
                 path);
    apothem *= 0.998929882193;
  }
  return 0;
}
