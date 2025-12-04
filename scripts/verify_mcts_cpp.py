import sys
import os
import chess
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.resnet import ChessResNet
from src.model.model_utils import get_device
from src.chess_utils import encode_board, ACTION_CONVERTER

def verify_mcts_cpp():
    print("--- Verifying C++ MCTS ---")
    
    try:
        import chess_engine_cpp
        print("Successfully imported chess_engine_cpp")
    except ImportError as e:
        print(f"Failed to import chess_engine_cpp: {e}")
        return

    device = get_device()
    model = ChessResNet(num_blocks=2, num_filters=32).to(device)
    model.eval()
    
    # Wrapper for model to accept numpy and return (p, v)
    def predict_fn(state_array):
        tensor = torch.from_numpy(state_array).unsqueeze(0).to(device)
        with torch.no_grad():
            p, v = model(tensor)
        return p.cpu().numpy()[0], v.item()
        
    # Initialize MCTS
    mcts = chess_engine_cpp.MCTS(predict_fn, 10, 1.0)
    
    # Search
    board = chess.Board()
    print("Starting search...")
    best_move, visits = mcts.search(board.fen())
    
    print(f"Best Move: {best_move}")
    print(f"Visits: {visits[:5]}...") # Show top 5
    
    print("Verification Complete.")

if __name__ == "__main__":
    verify_mcts_cpp()
