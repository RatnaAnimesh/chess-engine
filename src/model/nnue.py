import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

class NNUE(nn.Module):
    """
    NNUE (Efficiently Updatable Neural Network) Architecture.
    Adapted for PyTorch:
    - Input: Sparse Indices (HalfKP)
    - Layer 1: Feature Transformer (Linear)
    - Layer 2+: Fully Connected
    - Output: Policy (4672) + Value (1)
    """
    def __init__(self, feature_dim=41024, hidden_dim=256):
        super(NNUE, self).__init__()
        
        # Feature Transformer
        # In C++, this is updated incrementally. In PyTorch, we simulate it with a Linear layer.
        # Input dimension is large (HalfKP ~ 41k)
        self.feature_dim = feature_dim
        self.input_layer = nn.Linear(feature_dim, hidden_dim)
        
        # Hidden Layers (Clipped ReLU)
        self.l1 = nn.Linear(hidden_dim, 32)
        self.l2 = nn.Linear(32, 32)
        
        # Heads
        self.policy_head = nn.Linear(32, 4672)
        self.value_head = nn.Linear(32, 1)
        
    def forward(self, x):
        # x: (Batch, FeatureDim) - Sparse or Dense?
        # For training in PyTorch, we usually pass dense tensors or sparse tensors.
        # Given the size (41k), dense is memory heavy (41k * 4 bytes * Batch).
        # Batch 100 * 41k * 4 = 16MB. It's manageable.
        
        x = self.input_layer(x)
        
        # Clipped ReLU (0 to 1.0 in float, equivalent to 0-127 int8)
        x = torch.clamp(x, 0.0, 1.0)
        
        x = self.l1(x)
        x = torch.clamp(x, 0.0, 1.0)
        
        x = self.l2(x)
        x = torch.clamp(x, 0.0, 1.0)
        
        # Policy
        p = self.policy_head(x)
        
        # Value
        v = self.value_head(x)
        v = torch.tanh(v)
        
        return p, v

def encode_halfkp(board: chess.Board):
    """
    Encodes board into HalfKP feature vector (Dense for PyTorch).
    This is a simplified version.
    Real HalfKP: King Square (64) * Piece (10) * Square (64) = 40960
    + Pawns?
    
    Let's use a simplified sparse feature set for this demonstration:
    - Piece * Square (12 * 64 = 768)
    - King * Piece * Square is too big for simple dense training on laptop without sparse support.
    
    We will use Piece-Square (768) features for the Student to demonstrate the architecture.
    """
    features = np.zeros(768, dtype=np.float32)
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            # Piece type: 1-6 (P, N, B, R, Q, K)
            # Color: 0 (White), 1 (Black)
            # Index: (Color * 6 + (Type - 1)) * 64 + Square
            
            p_type = piece.piece_type - 1
            color = int(piece.color)
            
            idx = (color * 6 + p_type) * 64 + sq
            features[idx] = 1.0
            
    return features
