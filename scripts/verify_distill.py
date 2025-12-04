import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.resnet import ChessResNet
from src.model.nnue import NNUE
from src.model.model_utils import get_device
from src.training.dataset import ReplayBuffer, ChessDataset
from src.training.distill import DistillationTrainer
from src.self_play import run_self_play_loop

def verify_distill():
    print("--- Verifying Distillation ---")
    device = get_device()
    print(f"Device: {device}")
    
    # 1. Setup Teacher and Student
    teacher = ChessResNet(num_blocks=2, num_filters=32).to(device)
    student = NNUE(feature_dim=768, hidden_dim=64).to(device) # Tiny student
    
    # 2. Generate Data (Teacher plays)
    print("Generating data...")
    examples = run_self_play_loop(teacher, num_games=1, num_simulations=10)
    
    dataset = ChessDataset(examples)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 3. Distill
    print("Distilling...")
    trainer = DistillationTrainer(student, teacher, device)
    
    metrics = trainer.train_epoch(dataloader)
    print(f"Distillation Metrics: {metrics}")
    
    # Check if student learned something (sanity check)
    # Just check if it runs without error
    print("Verification Complete.")

if __name__ == "__main__":
    verify_distill()
