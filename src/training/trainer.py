import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from .dataset import ChessDataset

class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()
        
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        # policy_pred: (B, 4672) - Logits
        # value_pred: (B, 1) - Tanh output
        # policy_target: (B, 4672) - Probabilities
        # value_target: (B,) - Scalar {-1, 0, 1}
        
        # Value Loss: MSE
        # (z - v)^2
        value_loss = (value_target.view(-1, 1) - value_pred) ** 2
        value_loss = value_loss.mean()
        
        # Policy Loss: Cross Entropy
        # - sum(pi * log(p))
        # We use LogSoftmax + NLLLoss or just sum manually
        # policy_pred are logits.
        log_probs = nn.functional.log_softmax(policy_pred, dim=1)
        policy_loss = -torch.sum(policy_target * log_probs, dim=1).mean()
        
        return value_loss + policy_loss, value_loss, policy_loss

class Trainer:
    def __init__(self, model, device, learning_rate=0.02, weight_decay=1e-4):
        self.model = model
        self.device = device
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        self.criterion = AlphaZeroLoss()
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_v_loss = 0
        total_p_loss = 0
        batches = 0
        
        for state, _, policy, value in dataloader:
            state = state.to(self.device)
            policy = policy.to(self.device)
            value = value.to(self.device)
            
            self.optimizer.zero_grad()
            
            p_pred, v_pred = self.model(state)
            
            loss, v_loss, p_loss = self.criterion(p_pred, v_pred, policy, value)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_v_loss += v_loss.item()
            total_p_loss += p_loss.item()
            batches += 1
            
        return {
            'loss': total_loss / batches,
            'value_loss': total_v_loss / batches,
            'policy_loss': total_p_loss / batches
        }
        
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
    def load_checkpoint(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint from {path}")
        else:
            print(f"No checkpoint found at {path}")
