import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.timesnet_layers import TimesBlock

class TimesNetModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # 모델 파라미터
        self.d_model = configs.get('d_model', 512)
        self.e_layers = configs.get('e_layers', 2)
        self.d_ff = configs.get('d_ff', 2048)
        self.top_k = configs.get('top_k', 2)
        self.num_kernels = configs.get('num_kernels', 6)
        self.dropout = configs.get('dropout', 0.1)
        
        # 시계열 데이터와 시간 특성 분리를 위한 크기 저장
        self.data_feature_size = configs['num_features'] if self.mode == 'CD' else 1
        
        # 입력 프로젝션 (시계열 데이터만 처리)
        self.input_projection = nn.Linear(self.data_feature_size, self.d_model)
        
        # Times blocks
        self.layers = nn.ModuleList([
            TimesBlock(
                input_size=self.d_model,
                hidden_size=self.d_ff,
                top_k=self.top_k,
                num_kernels=self.num_kernels
            ) for _ in range(self.e_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_model)
        
        # 중요 수정 부분: 예측을 위한 선형 레이어를 변경
        # 전체 시퀀스 정보를 활용하여 pred_len 길이의 예측값 생성
        self.pred_network = nn.Sequential(
            nn.Linear(self.d_model * self.input_len, self.pred_len * self.output_size),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 시계열 데이터와 시간 특성 분리
        x_data = x[:, :, :self.data_feature_size]  # 시계열 데이터만 선택
        
        # 입력 정규화 (시계열 데이터만)
        means = x_data.mean(1, keepdim=True).detach()
        x_data = x_data - means
        stdev = torch.sqrt(torch.var(x_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_data = x_data / stdev
        
        # 입력 프로젝션
        x = self.input_projection(x_data)
        
        # Times blocks
        for layer in self.layers:
            x = self.norm(layer(x))
        
        # 수정된 부분: 시퀀스 전체를 사용하여 예측값 생성
        x = x.reshape(batch_size, -1)  # [batch_size, input_len * d_model]
        x = self.pred_network(x)  # [batch_size, pred_len * output_size]
        forecast = x.reshape(batch_size, self.pred_len, self.output_size)  # [batch_size, pred_len, output_size]
        
        # 예측값 정규화 복원
        forecast = forecast * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        forecast = forecast + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return forecast

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)