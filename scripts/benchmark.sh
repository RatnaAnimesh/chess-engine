#!/bin/bash

# Path to your engine
ENGINE_CMD="python3 -m src.main"
ENGINE_DIR="$(dirname "$0")/.."

# Path to opponent (Stockfish) - Assumes stockfish is in PATH
OPPONENT_CMD="stockfish"

# Time control: 40 moves in 60 seconds
TC="40/60"

# Number of games
GAMES=10

echo "Starting benchmark..."
echo "Engine: AppleSiliconChess ($ENGINE_CMD)"
echo "Opponent: Stockfish"
echo "Time Control: $TC"
echo "Games: $GAMES"

# Activate venv if needed
source "$ENGINE_DIR/venv/bin/activate"

cutechess-cli \
    -engine cmd="$ENGINE_CMD" dir="$ENGINE_DIR" name="AppleSiliconChess" proto=uci \
    -engine cmd="$OPPONENT_CMD" name="Stockfish" proto=uci \
    -each tc=$TC \
    -games $GAMES \
    -repeat \
    -pgnout benchmark_results.pgn \
    -concurrency 1

echo "Benchmark complete. Results saved to benchmark_results.pgn"
