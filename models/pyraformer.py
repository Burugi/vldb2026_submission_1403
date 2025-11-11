import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.pyraformer_layers import Encoder


class PyraformerModel(BaseModel):
    """ 
    Pyraformer: Pyramidal attention to reduce complexity
    Paper link: https://openreview.net/pdf?id=0EXmFzUn5I
    """

    def __init__(self, configs):
        """
        Args:
            configs (dict): 모델 설정
        """
        super().__init__(configs)
        
        # 기본 설정
        self.d_model = configs.get('d_model', 512)
        self.d_ff = configs.get('d_ff', 2048)
        self.n_heads = configs.get('n_heads', 8)
        self.e_layers = configs.get('e_layers', 2)
        self.dropout = configs.get('dropout', 0.1)
        
        # Pyraformer 설정
        self.window_size = configs.get('window_size', [4, 4])
        self.inner_size = configs.get('inner_size', 5)
        
        # 모델 구성 요소
        self.encoder = Encoder(
            d_model=self.d_model,
            d_ff=self.d_ff,
            seq_len=self.input_len,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            dropout=self.dropout,
            enc_in=self.output_size,  # output_size는 BaseModel에서 설정됨
            window_size=self.window_size,
            inner_size=self.inner_size
        )
        
        # 투영층
        self.projection = nn.Linear(
            (len(self.window_size) + 1) * self.d_model, 
            self.pred_len * self.output_size
        )
    
    def forward(self, x):
        """
        Args:
            x: 입력 텐서 [batch_size, input_len, input_size]
            
        Returns:
            예측 텐서 [batch_size, pred_len, output_size]
        """
        if self.mode == 'CD':
            return self._forward_cd(x)
        else:
            return self._forward_ci(x)
    
    def _forward_cd(self, x):
        """
        CD 모드: 다중 특성 예측
        """
        batch_size = x.shape[0]
        
        # 데이터와 시간 특성 분리
        features = x[:, :, :self.output_size]
        time_features = x[:, :, self.output_size:]
        
        # 정규화
        means = features.mean(1, keepdim=True).detach()
        features_norm = features - means
        stdev = torch.sqrt(torch.var(features_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        features_norm /= stdev
        
        # 인코더 통과
        enc_out = self.encoder(features_norm, time_features)[:, -1, :]
        
        # 투영 (예측 길이와 피처 수로 조정)
        output = self.projection(enc_out).view(batch_size, self.pred_len, self.output_size)
        
        # 역정규화
        output = output * stdev + means
        
        return output
    
    def _forward_ci(self, x):
        """
        CI 모드: 단일 특성 예측
        """
        batch_size = x.shape[0]
        
        # 데이터와 시간 특성 분리 (CI 모드에서는 첫 번째 열만 사용)
        feature = x[:, :, 0:1]
        time_features = x[:, :, 1:]
        
        # 정규화
        mean = feature.mean(1, keepdim=True).detach()
        feature_norm = feature - mean
        stdev = torch.sqrt(torch.var(feature_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        feature_norm /= stdev
        
        # 인코더 통과
        enc_out = self.encoder(feature_norm, time_features)[:, -1, :]
        
        # 투영 (예측 길이로 조정, 출력은 단일 특성)
        output = self.projection(enc_out).view(batch_size, self.pred_len, 1)
        
        # 역정규화
        output = output * stdev + mean
        
        return output