#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <string>

// This concepts defines the necessary functions a game must implement in order
// to connect to tourney. We use a C++20 concept rather than virtual base class
// to completely eliminate the cost of virtual dispatch.
template <typename T>
concept GameState =
    requires(T s, const T cs, typename T::MoveT m, typename T::MoveT* buf,
             std::size_t cap, const std::string input) {
      typename T::MoveT;

      { cs.FillLegalMoves(buf, cap) } -> std::same_as<std::size_t>;
      { s.MakeMove(m) } -> std::same_as<void>;
      { s.UnmakeMove(m) } -> std::same_as<void>;
      { cs.IsOver() } -> std::same_as<bool>;
      { cs.ToString() } -> std::same_as<std::string>;
      { cs.Parse(input) } -> std::same_as<std::optional<typename T::MoveT>>;
    };
