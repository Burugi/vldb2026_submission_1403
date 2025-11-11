import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueEmbedding(nn.Module):
    """시계열 데이터를 임베딩하는 레이어"""
    def __init__(self, seg_len, d_model):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(seg_len, d_model),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.embedding(x)

class RNNEncoder(nn.Module):
    """GRU 기반 인코더"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x, h=None):
        return self.rnn(x, h)

class PositionalChannelEmbedding(nn.Module):
    """위치 및 채널 임베딩"""
    def __init__(self, seg_num_y, enc_in, d_model):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even"
        d_half = d_model // 2
        
        # 위치 및 채널 임베딩 파라미터 초기화
        self.pos_emb = nn.Parameter(torch.randn(seg_num_y, d_half))
        self.channel_emb = nn.Parameter(torch.randn(enc_in, d_half))
        
        self.seg_num_y = seg_num_y
        self.enc_in = enc_in
        self.d_model = d_model
        
    def forward(self, batch_size):
        # 위치 임베딩 확장: [seg_num_y, d/2] -> [enc_in, seg_num_y, d/2]
        pos_emb = self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1)
        
        # 채널 임베딩 확장: [enc_in, d/2] -> [enc_in, seg_num_y, d/2]
        channel_emb = self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        
        # 위치와 채널 임베딩 결합
        combined_emb = torch.cat([pos_emb, channel_emb], dim=-1)
        
        # 배치 차원 추가
        combined_emb = combined_emb.view(-1, 1, self.d_model).repeat(batch_size, 1, 1)
        
        return combined_emb