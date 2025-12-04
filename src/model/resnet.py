import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = x.view(batch, channels, -1).mean(dim=2)
        # Excitation: Fully Connected -> ReLU -> Fully Connected -> Sigmoid
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Scale
        y = y.view(batch, channels, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with optional SE.
    """
    def __init__(self, channels, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class ChessResNet(nn.Module):
    """
    ResNet for Chess AlphaZero/Lc0 style.
    Input: (B, 119, 8, 8) - Standard AlphaZero input representation
    Output:
        - Policy: (B, 1968) - Move probabilities (simplified 73*8*8 flattened or similar)
          Actually, standard AlphaZero is 8x8x73 = 4672.
          Let's stick to a simplified output for now or the full 4672.
          We will use 4672 (73 planes of 8x8).
        - Value: (B, 1) - Scalar evaluation [-1, 1] or [0, 1]
    """
    def __init__(self, num_blocks=10, num_filters=128, input_channels=119, policy_channels=73):
        super(ChessResNet, self).__init__()
        
        # Initial Convolution
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        # Residual Tower
        self.res_tower = nn.Sequential(
            *[ResidualBlock(num_filters, use_se=True) for _ in range(num_blocks)]
        )

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 8 * 8 * policy_channels) # 4672

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Input
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu(x)

        # Tower
        x = self.res_tower(x)

        # Policy Head
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Note: Softmax is usually applied in the loss function or during inference selection

        # Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v) # Value is typically [-1, 1]

        return p, v
