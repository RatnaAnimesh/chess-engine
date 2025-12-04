import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .dataset import ChessDataset

class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse = nn.MSELoss()
        
    def forward(self, student_p, student_v, teacher_p, teacher_v):
        # Policy: KL Divergence
        # Teacher policy is already probabilities (or logits?)
        # If teacher_p are logits, we use log_softmax(student) and softmax(teacher)
        
        # We assume inputs are Logits
        student_log_probs = F.log_softmax(student_p / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_p / self.temperature, dim=1)
        
        policy_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Value: MSE
        value_loss = self.mse(student_v, teacher_v)
        
        return policy_loss + value_loss, policy_loss, value_loss

class DistillationTrainer:
    def __init__(self, student, teacher, device, learning_rate=0.01):
        self.student = student
        self.teacher = teacher
        self.device = device
        self.optimizer = optim.Adam(student.parameters(), lr=learning_rate)
        self.criterion = DistillationLoss()
        
    def train_epoch(self, dataloader):
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        batches = 0
        
        for state, sparse, _, _ in dataloader:
            # state: Dense (Teacher)
            # sparse: Sparse (Student)
            # We ignore stored targets, we use Teacher as target
            
            state = state.to(self.device)
            sparse = sparse.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_p, teacher_v = self.teacher(state)
                
            # Student forward
            student_p, student_v = self.student(sparse)
            
            # Loss
            loss, p_loss, v_loss = self.criterion(student_p, student_v, teacher_p, teacher_v)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            batches += 1
            
        return {
            'loss': total_loss / batches,
            'policy_loss': total_p_loss / batches,
            'value_loss': total_v_loss / batches
        }
