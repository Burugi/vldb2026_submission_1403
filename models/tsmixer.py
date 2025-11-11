import torch
import torch.nn as nn
from models.base_model import BaseModel
from layers.tsmixer_layers import ResidualBlock

class TSMixerModel(BaseModel):
    def __init__(self, configs):
        """
        Args:
            configs: 모델 설정값을 담은 딕셔너리
                - input_len: 입력 시퀀스 길이
                - pred_len: 예측 시퀀스 길이
                - mode: 'CD' 또는 'CI'
                - num_features: 특성 수
                - time_feature_size: 시간 특성 차원
                - d_model: mixing dimension
                - e_layers: residual block 수
                - dropout: dropout 비율
        """
        super().__init__(configs)
        
        # 시계열 데이터와 시간 특성 분리
        self.data_feature_size = configs['num_features'] if self.mode == 'CD' else 1
        
        # 모델 파라미터
        self.d_model = configs.get('d_model', 256)
        self.e_layers = configs.get('e_layers', 2)
        self.dropout = configs.get('dropout', 0.1)
        
        # Residual Blocks
        self.mixer_blocks = nn.ModuleList([
            ResidualBlock(
                seq_len=self.input_len,
                feature_size=self.data_feature_size,
                d_model=self.d_model,
                dropout=self.dropout
            ) for _ in range(self.e_layers)
        ])
        
        # 예측을 위한 프로젝션 레이어
        self.projection = nn.Linear(self.input_len, self.pred_len)
        
    def forward(self, x):
        """
        Args:
            x: 입력 텐서, shape [batch_size, input_len, input_size]
            
        Returns:
            예측값, shape [batch_size, pred_len, output_size]
        """
        # 시계열 데이터 선택
        x_data = x[:, :, :self.data_feature_size]
        
        # 정규화
        means = x_data.mean(1, keepdim=True).detach()
        x_data = x_data - means
        stdev = torch.sqrt(torch.var(x_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_data = x_data / stdev
        
        # Mixer blocks 적용
        for block in self.mixer_blocks:
            x_data = block(x_data)
            
        # 시퀀스 길이 변환
        x_data = self.projection(x_data.transpose(1, 2)).transpose(1, 2)
        
        # 정규화 복원
        x_data = x_data * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        x_data = x_data + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return x_data
    
    def configure_optimizers(self):
        """옵티마이저 설정"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)