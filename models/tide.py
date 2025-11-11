import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.tide_layers import ResBlock, LayerNorm

class TiDEModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # 기본 설정
        self.feature_encode_dim = configs.get('feature_encode_dim', 2)
        self.hidden_dim = configs.get('d_model', 128)
        self.res_hidden = configs.get('d_model', 128)
        self.encoder_num = configs.get('e_layers', 2)
        self.decoder_num = configs.get('d_layers', 2)
        self.temporal_decoder_hidden = configs.get('d_ff', 256)
        self.dropout = configs.get('dropout', 0.1)
        self.bias = configs.get('bias', True)
        
        # 입력 차원 및 특성 관련 설정
        self.time_dim = configs.get('time_feature_size', 4)
        
        # CD/CI 모드에 따른 설정
        if self.mode == 'CD':
            self.num_features = configs['num_features']
            self.decode_dim = self.num_features
        else:  # CI mode
            self.num_features = 1
            self.decode_dim = 1
            
        # 입력 차원 계산
        self.flatten_dim = self.input_len + (self.input_len + self.pred_len) * self.feature_encode_dim
        
        # 모델 레이어 정의
        # 특성 인코더
        self.feature_encoder = ResBlock(
            self.time_dim, 
            self.res_hidden, 
            self.feature_encode_dim, 
            self.dropout, 
            self.bias
        )
        
        # 인코더 레이어
        encoder_layers = [
            ResBlock(self.flatten_dim, self.res_hidden, self.hidden_dim, self.dropout, self.bias)
        ]
        encoder_layers.extend([
            ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, self.dropout, self.bias)
            for _ in range(self.encoder_num - 1)
        ])
        self.encoders = nn.Sequential(*encoder_layers)
        
        # 디코더 레이어
        decoder_layers = []
        decoder_layers.extend([
            ResBlock(self.hidden_dim, self.res_hidden, self.hidden_dim, self.dropout, self.bias)
            for _ in range(self.decoder_num - 1)
        ])
        decoder_layers.append(
            ResBlock(self.hidden_dim, self.res_hidden, self.decode_dim * self.pred_len, self.dropout, self.bias)
        )
        self.decoders = nn.Sequential(*decoder_layers)
        
        # 시간적 디코더
        self.temporal_decoder = ResBlock(
            self.decode_dim + self.feature_encode_dim, 
            self.temporal_decoder_hidden, 
            1, 
            self.dropout, 
            self.bias
        )
        
        # 잔차 연결을 위한 프로젝션
        self.residual_proj = nn.Linear(self.input_len, self.pred_len, bias=self.bias)
        
    def forward(self, x):
        """
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
        batch_size = x.shape[0]
        
        # 시계열 데이터와 시간 특성 분리
        features = x[:, :, :self.num_features]
        time_features = x[:, :, self.num_features:]
        
        # 정규화
        means = features.mean(1, keepdim=True).detach()
        features_norm = features - means
        stdev = torch.sqrt(torch.var(features_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        features_norm /= stdev
        
        # 가상의 미래 시간 특성 생성 (실제 애플리케이션에서는 이 부분이 제공될 수 있음)
        future_time_features = torch.zeros((batch_size, self.pred_len, self.time_dim), device=x.device)
        all_time_features = torch.cat([time_features, future_time_features], dim=1)
        
        # 시간 특성 인코딩
        encoded_time_features = self.feature_encoder(all_time_features)
        encoded_time_features_flat = encoded_time_features.reshape(batch_size, -1)
        
        # 전체 예측 결과 저장
        all_predictions = []
        
        # 각 특성별로 예측 수행
        for feature_idx in range(self.num_features):
            # 인코더 입력 준비
            feature_data = features_norm[:, :, feature_idx:feature_idx+1].squeeze(-1)
            encoder_input = torch.cat([feature_data, encoded_time_features_flat], dim=-1)
            
            # 인코더-디코더 통과
            hidden = self.encoders(encoder_input)
            decoded = self.decoders(hidden).reshape(batch_size, self.pred_len, self.decode_dim)
            
            # 시간적 디코딩 및 잔차 연결
            prediction = self.temporal_decoder(
                torch.cat([encoded_time_features[:, self.input_len:], decoded], dim=-1)
            ).squeeze(-1) + self.residual_proj(feature_data)
            
            # 결과 저장
            all_predictions.append(prediction)
        
        # 모든 특성의 예측을 결합
        all_predictions = torch.stack(all_predictions, dim=-1)
        
        # 역정규화
        all_predictions = all_predictions * stdev.squeeze(1).unsqueeze(1)
        all_predictions = all_predictions + means.squeeze(1).unsqueeze(1)
        
        return all_predictions
    
    def _forward_ci(self, x):
        """CI 모드: 단일 특성 예측"""
        batch_size = x.shape[0]
        
        # 시계열 데이터와 시간 특성 분리 (CI 모드에서는 첫 번째 차원만 사용)
        features = x[:, :, 0:1]
        time_features = x[:, :, 1:]
        
        # 정규화
        means = features.mean(1, keepdim=True).detach()
        features_norm = features - means
        stdev = torch.sqrt(torch.var(features_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        features_norm /= stdev
        
        # 가상의 미래 시간 특성 생성 (실제 애플리케이션에서는 이 부분이 제공될 수 있음)
        future_time_features = torch.zeros((batch_size, self.pred_len, self.time_dim), device=x.device)
        all_time_features = torch.cat([time_features, future_time_features], dim=1)
        
        # 시간 특성 인코딩
        encoded_time_features = self.feature_encoder(all_time_features)
        encoded_time_features_flat = encoded_time_features.reshape(batch_size, -1)
        
        # 인코더 입력 준비
        feature_data = features_norm.squeeze(-1)
        encoder_input = torch.cat([feature_data, encoded_time_features_flat], dim=-1)
        
        # 인코더-디코더 통과
        hidden = self.encoders(encoder_input)
        decoded = self.decoders(hidden).reshape(batch_size, self.pred_len, self.decode_dim)
        
        # 시간적 디코딩 및 잔차 연결
        prediction = self.temporal_decoder(
            torch.cat([encoded_time_features[:, self.input_len:], decoded], dim=-1)
        ).squeeze(-1) + self.residual_proj(feature_data)
        
        # 차원 조정
        prediction = prediction.unsqueeze(-1)
        
        # 역정규화
        prediction = prediction * stdev
        prediction = prediction + means
        
        return prediction