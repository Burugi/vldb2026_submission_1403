import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base_model import BaseModel
from layers.scinet_layers import SCINetTree

class SCINetModel(BaseModel):
    def __init__(self, configs):
        """
        Args:
            configs: 모델 설정값을 담은 딕셔너리
                - input_len: 입력 시퀀스 길이
                - pred_len: 예측 시퀀스 길이
                - mode: 'CD' 또는 'CI'
                - num_features: 특성 수
                - time_feature_size: 시간 특성 차원
                - levels: SCINet tree의 레벨 수
                - kernel_size: 컨볼루션 커널 크기
                - dropout: dropout 비율
                - stacks: SCINet 스택 수 (1 또는 2)
        """
        super().__init__(configs)
        
        # 시계열 데이터와 시간 특성 분리
        self.data_feature_size = configs['num_features'] if self.mode == 'CD' else 1
        
        # 모델 파라미터
        self.levels = configs.get('levels', 3)
        self.kernel_size = configs.get('kernel_size', 5)
        self.dropout = configs.get('dropout', 0.1)
        self.stacks = configs.get('stacks', 1)
        
        # SCINet 스택 구성
        self.scinet_stack1 = SCINetTree(
            d_model=self.data_feature_size,
            current_level=self.levels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        
        if self.stacks == 2:
            self.scinet_stack2 = SCINetTree(
                d_model=self.data_feature_size,
                current_level=self.levels,
                kernel_size=self.kernel_size,
                dropout=self.dropout
            )
            
        # 예측을 위한 컨볼루션 레이어
        self.projection1 = nn.Conv1d(
            self.input_len, 
            self.input_len + self.pred_len, 
            kernel_size=1,
            stride=1,
            bias=False
        )
        
        if self.stacks == 2:
            self.projection2 = nn.Conv1d(
                self.input_len + self.pred_len,
                self.input_len + self.pred_len,
                kernel_size=1,
                bias=False
            )
            
    def forward(self, x):
        """
        Args:
            x: 입력 텐서, shape [batch_size, input_len, input_size]
            
        Returns:
            예측값, shape [batch_size, pred_len, output_size]
        """
        batch_size = x.size(0)
        
        # 시계열 데이터 선택
        x_data = x[:, :, :self.data_feature_size]
        
        # 정규화
        means = x_data.mean(1, keepdim=True).detach()
        x_data = x_data - means
        stdev = torch.sqrt(torch.var(x_data, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_data = x_data / stdev
        
        # 첫 번째 스택 적용
        out = self.scinet_stack1(x_data)
        out = out + x_data  # 잔차 연결
        out = self.projection1(out)
        
        # 두 번째 스택 적용 (있는 경우)
        if self.stacks == 2:
            # 이전 입력과 결합
            out = torch.cat([x_data, out], dim=1)
            temp = out
            
            out = self.scinet_stack2(out)
            out = out + temp  # 잔차 연결
            out = self.projection2(out)
        
        # 정규화 복원
        out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, out.size(1), 1))
        out = out + (means[:, 0, :].unsqueeze(1).repeat(1, out.size(1), 1))
        
        # 예측 구간만 선택
        return out[:, -self.pred_len:, :]

    def configure_optimizers(self):
        """옵티마이저 설정"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)