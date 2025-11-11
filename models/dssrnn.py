import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from models.ssrnn import SSRNNModel
from layers.dssrnn_layers import SeriesDecomposition

class DSSRNNModel(BaseModel):
    """
    Decomposed State Space RNN (DSSRNN) Model
    시계열을 계절성(Seasonal)과 추세(Trend) 성분으로 분해하여 처리하는 모델
    """
    def __init__(self, configs):
        super(DSSRNNModel, self).__init__(configs)
        
        # 분해 커널 크기 설정
        self.kernel_size = configs.get('kernel_size', 25)
        self.decomposition = SeriesDecomposition(self.kernel_size)
        
        # 내부에서 사용할 SSRNN 모델 및 선형 레이어 초기화
        ssrnn_configs = configs.copy()
        self.SSRNN_Seasonal = SSRNNModel(ssrnn_configs)
        self.Linear_Trend = nn.Linear(self.input_len, self.pred_len)
            
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
        
        # 시계열 분해
        seasonal_init, trend_init = self.decomposition(features_norm)
        
        # 추세 성분 처리를 위한 차원 변환
        trend_init = trend_init.permute(0, 2, 1)  # [batch, channel, seq_len]
        
        # 계절성 성분 예측
        seasonal_output = self.SSRNN_Seasonal(seasonal_init)
        
        # 추세 성분 예측 (선형 레이어 사용)
        trend_output = self.Linear_Trend(trend_init)  # [batch, channel, pred_len]
        trend_output = trend_output.permute(0, 2, 1)  # [batch, pred_len, channel]
        
        # 예측 결합
        x = seasonal_output + trend_output
        
        # 역정규화
        x = x * stdev + means
        
        return x
    
    def _forward_ci(self, x):
        """CI 모드: 단일 특성 예측"""
        # 데이터와 시간 특성 분리
        feature = x[:, :, 0:1]  # 첫 번째 특성만 사용
        
        # 정규화
        mean = feature.mean(1, keepdim=True).detach()
        feature_norm = feature - mean
        stdev = torch.sqrt(torch.var(feature_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        feature_norm /= stdev
        
        # 시계열 분해
        seasonal_init, trend_init = self.decomposition(feature_norm)
        
        # 추세 성분 처리를 위한 차원 변환
        trend_init = trend_init.permute(0, 2, 1)  # [batch, 1, seq_len]
        
        # 계절성 성분 예측
        seasonal_output = self.SSRNN_Seasonal(seasonal_init)
        
        # 추세 성분 예측 (선형 레이어 사용)
        trend_output = self.Linear_Trend(trend_init)  # [batch, 1, pred_len]
        trend_output = trend_output.permute(0, 2, 1)  # [batch, pred_len, 1]
        
        # 예측 결합
        x = seasonal_output + trend_output
        
        # 역정규화
        x = x * stdev + mean
        
        return x