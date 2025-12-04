import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, example):
        """
        example: (state, sparse, policy, value)
        """
        self.buffer.append(example)
        
    def extend(self, examples):
        self.buffer.extend(examples)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class ChessDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        state, sparse, policy, value = self.examples[idx]
        
        state_tensor = torch.from_numpy(state).float()
        sparse_tensor = torch.from_numpy(sparse).float()
        policy_tensor = torch.from_numpy(policy).float()
        value_tensor = torch.tensor(value, dtype=torch.float32)
        
        return state_tensor, sparse_tensor, policy_tensor, value_tensor
