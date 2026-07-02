#include <print>
#include <string>

#define MAX_VERBOSITY 3
#define VLOG(verbosity) Logger(__FILE__, __LINE__, verbosity)

// TODO This should probably go in some other utility directory, or maybe just
// switch to ABSL logging?
class Logger {
 private:
  const char *file_;
  int line_;
  int verbosity_;

 public:
  Logger(const char *file, int line, int verbosity)
      : file_(file), line_(line), verbosity_(verbosity) {}

  Logger &operator<<(const std::string &msg) {
    if (verbosity_ <= MAX_VERBOSITY) {
      std::println("[{}:{}]: {}", file_, line_, msg);
    }
    return *this;
  }
};
