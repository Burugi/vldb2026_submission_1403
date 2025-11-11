import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """시간 및 채널 방향의 믹싱을 수행하는 Residual Block"""
    def __init__(self, seq_len, feature_size, d_model, dropout=0.1):
        super().__init__()
        
        # 시간 방향 믹싱
        self.temporal_mixer = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout)
        )
        
        # 채널 방향 믹싱
        self.feature_mixer = nn.Sequential(
            nn.Linear(feature_size, d_model),
            nn.ReLU(),
            nn.Linear(d_model, feature_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: 입력 텐서, shape [batch_size, seq_len, feature_size]
        """
        # 시간 방향 믹싱
        temporal_out = self.temporal_mixer(x.transpose(1, 2)).transpose(1, 2)
        x = x + temporal_out
        
        # 채널 방향 믹싱
        feature_out = self.feature_mixer(x)
        x = x + feature_out
        
        return x