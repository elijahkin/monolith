# Tourney

Tourney is a library for implementing games as well as agents to play them. A
game (such as chess, tic-tac-toe, etc.) is added by satisfying the `GameState`.
In contrast, agents are the objects which play the game; for example,
`HumanPlayer` prompts the user for input, while `MinimaxPlayer` searches the
game tree for optimal moves.

[Minimax](https://en.wikipedia.org/wiki/Minimax) and its variations are a
well-known family of algorithms for implementing chess computers. The central
goal of Tourney is to make applying these algorithms to any two-player
adversarial game as simple as possible.

## Work in Progress

We divide work broadly into that which pertains specifically to chess and that
which does not.

### Chess-specific Work

- Implement the
  [fifty-move rule](https://en.wikipedia.org/wiki/Fifty-move_rule). Note:
  `ChessState::MakeMove` intentionally does not maintain `halfmove_clock_` (or
  `fullmove_number_`) on either the perft or interactive move-application path,
  since the values would go stale during search-tree traversal and no current
  caller relies on them after `MakeMove`. A future implementation of this rule
  will need to either restore clock updates on the interactive path or compute
  the halfmove counter from a recorded move history.
- Ensure `Parse` and `get_algebraic_notation` follow standard conventions and
  cleanly resolve ambiguities. We should probably add thorough testing for this.
- Build out the command line interface. It would be nice to have a move
  selection menu which uses arrow keys.

### General Work

- Add extensive testing (perhaps using a trivial game) of `MinimaxSearch`.
- Add benchmarking of `MinimaxSearch` in `chess_benchmark.cpp`. Once we can
  benchmark, we can then explore if switching to
  [negamax algorithm](https://en.wikipedia.org/wiki/Negamax) to avoid branching
  would be a meaningful improvement. We should also be able to configure whether
  to use alpha-beta pruning.
- Finally, we can then explore other avenues of improvement: transposition
  tables, quiescence search, iterative deepening, random optimal move selection.
- Make `history_` a vector of moves instead of a vector of strings. Then, if
  it's not _too_ expensive, we could use this to eliminate `RecordMove` (but
  would require recording the move even if it's not yet selected).
- Implement `RandomPlayer`, which selects uniformly from the set of possible
  moves.
- Design a tournament and ELO system to have many agents compete against each
  other.
- Implement other games: dots and boxes, 2048, blackjack, and poker.
- Consider whether declaring a `namespace tourney` would be appropriate.
- Address any reported clang-tidy issues. Judicious use of `// NOLINT` is
  permissible.
- Continuously rewrite code to make it as self-documenting as possible. Resort
  to comments sparingly. Ensure that "the function" is the implicit subject of
  all comments and that proper punctuation is used.

## References

- [Chess Programming Wiki](https://www.chessprogramming.org)
