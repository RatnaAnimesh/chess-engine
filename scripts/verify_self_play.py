import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.resnet import ChessResNet
from src.model.model_utils import get_device
from src.self_play import self_play_game

def verify_self_play():
    print("--- Verifying Self-Play Loop ---")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Initialize model
    model = ChessResNet(num_blocks=2, num_filters=32) # Tiny model for speed
    model.to(device)
    model.eval()
    
    print("Starting self-play game (simulations=10)...")
    # Low simulations for quick test
    examples = self_play_game(model, num_simulations=10, temp_threshold=5)
    
    print(f"Game finished. Generated {len(examples)} examples.")
    
    if len(examples) == 0:
        print("ERROR: No examples generated.")
        return
        
    # Check data integrity
    state, sparse, policy, value = examples[0]
    
    print(f"State Shape: {state.shape}")
    print(f"Policy Shape: {policy.shape}")
    print(f"Value: {value}")
    
    assert state.shape == (119, 8, 8)
    assert policy.shape == (4672,)
    assert isinstance(value, float)
    
    print("Verification Complete: Data format is correct.")

if __name__ == "__main__":
    verify_self_play()
