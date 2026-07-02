// Usage: bazel run //collatz 30

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>

int f(int n) { return (n % 2 == 0) ? (n >> 1) : ((3 * n) + 1); }

int main(int argc, char *argv[]) {
  // Retrieve maximum integer from command line
  const int N = std::atoi(argv[1]);

  // Determine output directory
  const char *workspace = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  const std::filesystem::path output_path =
      argc > 2    ? std::filesystem::path(argv[2])
      : workspace ? std::filesystem::path(workspace)
                  : std::filesystem::current_path();
  const std::string dir =
      (workspace && output_path.is_relative())
          ? (std::filesystem::path(workspace) / output_path).string()
          : output_path.string();
  std::filesystem::create_directories(dir);

  const std::string svg_path =
      (std::filesystem::path(dir) / std::format("collatz{}.svg", N)).string();
  const std::string gv_path = (std::filesystem::path(dir) / "tmp.gv").string();

  // Write to the .gv file
  std::ofstream file(gv_path);
  if (file.is_open()) {
    file << "digraph G {\n";

    for (int n = 1; n <= N; ++n) {
      file << std::format("  {} -> {};\n", n, f(n));
    }

    file << "}";
    file.close();
  }

  // Invoke GraphViz to generate visual from .gv file
  system(std::format("dot -Tsvg {} > {}", gv_path, svg_path).c_str());
  system(std::format("rm {}", gv_path).c_str());

  std::cout << "Wrote collatz output to file " << svg_path << '\n';
  return 0;
}
