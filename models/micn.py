import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.Embed import DataEmbedding
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi
from layers.micn_layers import MIC, SeasonalPrediction


class MICNModel(BaseModel):
    """
    MICN: Multi-scale Isometric Convolution Network for Time Series Forecasting
    
    Paper link: https://openreview.net/pdf?id=zt53IDUR1U
    """
    def __init__(self, configs):
        super().__init__(configs)
        
        # 모델 설정
        self.d_model = configs.get('d_model', 512)
        self.n_heads = configs.get('n_heads', 8)
        self.d_layers = configs.get('d_layers', 1)
        self.dropout = configs.get('dropout', 0.05)
        self.conv_kernel = configs.get('conv_kernel', [12, 16])
        
        # 합성곱 커널 및 등척성 커널 계산
        decomp_kernel = []  # 분해 연산의 커널
        isometric_kernel = []  # 등척성 합성곱의 커널
        for ii in self.conv_kernel:
            if ii % 2 == 0:  # 분해 연산의 커널은 반드시 홀수여야 함
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((self.input_len + self.pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((self.input_len + self.pred_len + ii - 1) // ii)
        
        # 다중 시리즈 분해 블록
        self.decomp_multi = series_decomp_multi(decomp_kernel)
        
        # 임베딩 레이어
        self.dec_embedding = DataEmbedding(
            self.output_size, self.d_model, 'timeF', 'h', self.dropout
        )
        
        # 계절성 예측 모듈
        self.conv_trans = SeasonalPrediction(
            embedding_size=self.d_model, 
            n_heads=self.n_heads,
            dropout=self.dropout,
            d_layers=self.d_layers, 
            decomp_kernel=decomp_kernel,
            c_out=self.output_size, 
            conv_kernel=self.conv_kernel,
            isometric_kernel=isometric_kernel, 
            device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        )
        
        # 트렌드 회귀를 위한 선형 레이어
        self.regression = nn.Linear(self.input_len, self.pred_len)
        self.regression.weight = nn.Parameter(
            (1 / self.pred_len) * torch.ones([self.pred_len, self.input_len]),
            requires_grad=True
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
        
        # 다중 스케일 하이브리드 분해
        seasonal_init_enc, trend = self.decomp_multi(features_norm)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 임베딩 및 합성곱 처리
        # 미래 시간 특성 생성 (0으로 채움)
        future_time_features = torch.zeros(
            [batch_size, self.pred_len, time_features.shape[-1]], 
            device=x.device
        )
        time_features_full = torch.cat([time_features, future_time_features], dim=1)
        
        # 계절성 컴포넌트 확장
        zeros = torch.zeros([batch_size, self.pred_len, self.output_size], device=x.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.input_len:, :], zeros], dim=1)
        
        # 임베딩 및 예측
        dec_out = self.dec_embedding(seasonal_init_dec, time_features_full)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        
        # 역정규화
        dec_out = dec_out * stdev + means
        
        return dec_out

    def _forward_ci(self, x):
        """
        CI 모드: 단일 특성 예측
        """
        batch_size = x.shape[0]
        
        # 데이터와 시간 특성 분리
        feature = x[:, :, 0:1]
        time_features = x[:, :, 1:]
        
        # 정규화
        mean = feature.mean(1, keepdim=True).detach()
        feature_norm = feature - mean
        stdev = torch.sqrt(torch.var(feature_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        feature_norm /= stdev
        
        # 다중 스케일 하이브리드 분해
        seasonal_init_enc, trend = self.decomp_multi(feature_norm)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 임베딩 및 합성곱 처리
        # 미래 시간 특성 생성 (0으로 채움)
        future_time_features = torch.zeros(
            [batch_size, self.pred_len, time_features.shape[-1]], 
            device=x.device
        )
        time_features_full = torch.cat([time_features, future_time_features], dim=1)
        
        # 계절성 컴포넌트 확장
        zeros = torch.zeros([batch_size, self.pred_len, 1], device=x.device)
        seasonal_init_dec = torch.cat([seasonal_init_enc[:, -self.input_len:, :], zeros], dim=1)
        
        # 임베딩 및 예측
        dec_out = self.dec_embedding(seasonal_init_dec, time_features_full)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len:, :] + trend[:, -self.pred_len:, :]
        
        # 역정규화
        dec_out = dec_out * stdev + mean
        
        return dec_out