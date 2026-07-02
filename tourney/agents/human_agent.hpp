#include <iostream>
#include <string>

#include "../contracts.hpp"

// Prompts the user for command line input, which is parsed via `state.Parse`.
// If the parse fails, alerts the user and prompts for input again.
template <typename Move>
class HumanAgent final : public Agent<Move> {
 public:
  explicit HumanAgent(Game<Move>& state) : Agent<Move>(state) {}

  Move SelectMove() override {
    std::string input;
    while (true) {
      std::cout << "Please enter a move: ";
      std::cin >> input;
      auto parsed = this->state_.Parse(input);
      if (parsed.has_value()) {
        return parsed.value();
        break;
      }
      std::cout << "Invalid entry! ";
    }
  }
};
