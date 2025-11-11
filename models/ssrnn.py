import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.ssrnn_layers import CustomRNNCell

class SSRNNModel(BaseModel):
    """
    State Space RNN (SSRNN) Model
    """
    def __init__(self, configs):
        super(SSRNNModel, self).__init__(configs)
        
        # 모델 파라미터
        self.hidden_size = configs.get('hidden_size', 128)
        self.num_layers = configs.get('num_layers', 1)
        
        # CD/CI 모드에 따른 채널 수 설정
        if self.mode == 'CD':
            self.channels = self.output_size  # 모든 특성
        else:  # CI mode
            self.channels = 1  # 단일 특성
            
        # RNN 셀 정의
        self.rnn_cell = CustomRNNCell(input_size=self.input_len, hidden_size=self.hidden_size)
            
        # 출력 레이어
        self.fc = nn.Linear(self.hidden_size, self.pred_len)
    
    def forward(self, x):
        """
        모델 순전파
        
        Args:
            x: 입력 텐서, shape [batch_size, input_len, input_size]
            
        Returns:
            출력 텐서, shape [batch_size, pred_len, output_size]
        """
        if self.mode == 'CD':
            return self._forward_cd(x)
        else:  # CI mode
            return self._forward_ci(x)
    
    def _forward_cd(self, x):
        """CD 모드: 모든 특성 예측"""
        # 데이터와 시간 특성 분리
        features = x[:, :, :self.output_size]
        
        # 정규화
        means = features.mean(1, keepdim=True).detach()
        features_norm = features - means
        stdev = torch.sqrt(torch.var(features_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        features_norm /= stdev
        
        # [Batch, Input length, Channel] -> [Batch, Channel, Input length]
        x_enc = features_norm.permute(0, 2, 1)
        batch_size = x_enc.size(0)
        
        h = torch.zeros(batch_size, self.hidden_size).to(x_enc.device)
        out = []
        for t in range(self.channels):
            h = self.rnn_cell(x_enc[:, t, :], h)
            out.append(h.unsqueeze(1))
        out = torch.cat(out, dim=1)
        
        # 예측 생성
        out = self.fc(out)  # [batch_size, channels, pred_len]
        out = out.permute(0, 2, 1)  # [batch_size, pred_len, channels]
        
        # 역정규화
        out = out * stdev + means
        
        return out
    
    def _forward_ci(self, x):
        """CI 모드: 단일 특성 예측"""
        # 데이터와 시간 특성 분리
        feature = x[:, :, 0:1]  # 첫 번째 특성만 사용
        
        # 정규화
        mean = feature.mean(1, keepdim=True).detach()
        feature_norm = feature - mean
        stdev = torch.sqrt(torch.var(feature_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        feature_norm /= stdev
        
        # [Batch, Input length, 1] -> [Batch, 1, Input length]
        x_enc = feature_norm.permute(0, 2, 1)
        batch_size = x_enc.size(0)
        
        # RNN 처리
        h = torch.zeros(batch_size, self.hidden_size).to(x_enc.device)
        out = self.rnn_cell(x_enc[:, 0, :], h)
        
        # 예측 생성
        out = self.fc(out).unsqueeze(-1)  # [batch_size, pred_len, 1]
        
        # 역정규화
        out = out * stdev + mean
        
        return out