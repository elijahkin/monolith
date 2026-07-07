#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <string>

// This concept defines the necessary functions a game must implement in order
// to connect to tourney. We use a C++20 concept rather than a virtual base
// class to completely eliminate the cost of virtual dispatch.

template <typename T>
concept GameState =
    requires(T s, const T cs, typename T::MoveT m, typename T::MoveUndo u,
             typename T::MoveT* buf, std::size_t cap, const std::string input) {
      // The game must define types `MoveT` and `MoveUndo`, which are used for
      // representing moves and "unmoves" within the game. An "unmove" holds any
      // information required for undoing the move which is not already
      // contained within the move itself. For exmaple, in chess, we need to
      // store a captured piece type or castling rights from before the move was
      // made. Itself is perfectly fine to define MoveUndo as empty struct if no
      // additioal information is needed beyond the move itself.
      typename T::MoveT;
      typename T::MoveUndo;

      // Populate every legal move into the provided buffer.
      { cs.FillLegalMoves(buf, cap) } -> std::same_as<std::size_t>;

      // Make the move and write undo information to the unmo out-param.
      { s.MakeMove(m, u) } -> std::same_as<void>;

      // Use the undo information to restore the state from before the move.
      { s.UnmakeMove(m, u) } -> std::same_as<void>;

      { cs.IsOver() } -> std::same_as<bool>;

      // Return a string representation of the game state, to be used for
      // debugging and/or processing by some user interface.
      { cs.ToString() } -> std::same_as<std::string>;

      { cs.Parse(input) } -> std::same_as<std::optional<typename T::MoveT>>;
    };
