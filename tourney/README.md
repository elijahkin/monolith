# Tourney

Tourney is an library for implementing games as well as agents to play them. The
fundamental classes are `Game` and `Agent`. A game (such as chess, go, etc.) can
be added by deriving from the former and implementing each of its virtual
functions. In contrast, agents are the objects which play the game; for example,
the `HumanAgent` prompts the user for input, while `MinimaxAgent` searches the
game tree for optimal moves.

[Minimax](https://en.wikipedia.org/wiki/Minimax) and its variations are a
well-known family of algorithms for implementing chess computers. The central
goal of Tourney is to make applying these algorithms to any two-player
adversarial game as simple as possible.

## Work in Progress

We divide work broadly into that which pertains specifically to `chess.hpp` and
that which does not.

### Chess-specific Work

- Refactor to use a
  [bitboard](https://en.wikipedia.org/wiki/Bitboard#Chess_bitboards)
  architecture for move generation. This will be tedious but (hopefully) worth
  the effort.
  - Upon switching to bitboards, use `__builtin_popcountll` based evaluation
    functions.
- Implement the
  [fifty-move rule](https://en.wikipedia.org/wiki/Fifty-move_rule).
- Implement special moves: pawn promotions, en passant capture, and castling.
  Ensure `Parse` and `GetAlgebraicNotation` are updated appropriately as well.
- Add unit tests to guarantee correctness.
- Fix problem with knight move generation.
- Generalize parsing of algebraic notation to allow disambiguation via
  specification of the `from` square.

### General Work

- Consider switching `MinimaxAgent` to use the
  [negamax algorithm](https://en.wikipedia.org/wiki/Negamax) to avoid branching.
- Make `history_` a vector of moves instead of a vector of strings. Then, if
  it's not _too_ expensive, we could use this to eliminate `RecordMove` (but
  would require recording the move even if it's not yet selected).
- Consider avenues of improvement for `MinimaxAgent`: iterative deepening,
  transposition tables, random optimal move selection
- Implement `RandomAgent`, which selects uniformly from the set of possible
  moves.
- Design a tournament and ELO system to have many agents compete against each
  other.
- Implement other games: dots and boxes, 2048, blackjack, and poker.
- Consider whether declaring a `namespace tourney` would be appropriate
- Address any reported clang-tidy issues. Judicious use of `// NOLINT` is
  permissible.
- Continuously rewrite code to make it as self-documenting as possible. Resort
  to comments sparingly. Ensure that "the function" is the implicit subject of
  all comments and that proper punctuation is used.

## References

- [Chess Programming Wiki](https://www.chessprogramming.org)
