import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import ProbAttention, AttentionLayer
from layers.informer_layers import Encoder, Decoder, EncoderLayer, DecoderLayer, ConvLayer

class InformerModel(BaseModel):
    """
    Informer: ProbSparse 어텐션을 통한 효율적인 시계열 예측 모델
    O(L log L) 복잡도를 가진 트랜스포머 기반 아키텍처
    
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super().__init__(configs)
        
        # 기본 파라미터 설정
        self.label_len = configs.get('label_len', 0)
        self.pred_len = configs.get('pred_len', 96)
        
        # 추가 파라미터
        self.d_model = configs.get('d_model', 512)
        self.d_ff = configs.get('d_ff', 2048)
        self.n_heads = configs.get('n_heads', 8)
        self.e_layers = configs.get('e_layers', 2)
        self.d_layers = configs.get('d_layers', 1)
        self.factor = configs.get('factor', 5)
        self.distil = configs.get('distil', True)
        self.dropout = configs.get('dropout', 0.1)
        self.activation = configs.get('activation', 'gelu')
        self.embed = configs.get('embed', 'timeF')
        self.freq = configs.get('freq', 'h')
        
        # Embedding
        self.enc_embedding = DataEmbedding(
            self.output_size, self.d_model, self.embed, self.freq, self.dropout
        )
        self.dec_embedding = DataEmbedding(
            self.output_size, self.d_model, self.embed, self.freq, self.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for _ in range(self.e_layers - 1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    AttentionLayer(
                        ProbAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for _ in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
            projection=nn.Linear(self.d_model, self.output_size, bias=True)
        )

    def forward(self, x):
        """
        Args:
            x: 입력 텐서 [batch_size, input_len, input_size]
            
        Returns:
            출력 텐서 [batch_size, pred_len, output_size]
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
        
        # 인코더 입력
        enc_out = self.enc_embedding(features_norm, time_features)
        
        # 디코더 입력 (예측할 시간 범위의 초기 값)
        # 디코더 입력은 레이블 길이 + 예측 길이
        dec_input = torch.zeros(
            (batch_size, self.pred_len, self.output_size),
            device=x.device
        )
        dec_input = torch.cat([features_norm[:, -self.label_len:, :], dec_input], dim=1)
        
        # 미래 시간 특성 생성 (0으로 채움)
        future_time_features = torch.zeros(
            (batch_size, self.pred_len, time_features.shape[-1]), 
            device=x.device
        )
        time_features_dec = torch.cat(
            [time_features[:, -self.label_len:, :], future_time_features], 
            dim=1
        )
        
        # 인코더 통과
        enc_out, _ = self.encoder(enc_out)
        
        # 디코더 임베딩
        dec_out = self.dec_embedding(dec_input, time_features_dec)
        
        # 디코더 통과
        dec_out = self.decoder(dec_out, enc_out)
        
        # 예측 부분만 추출
        dec_out = dec_out[:, -self.pred_len:, :]
        
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
        
        # 인코더 입력
        enc_out = self.enc_embedding(feature_norm, time_features)
        
        # 디코더 입력 (예측할 시간 범위의 초기 값)
        dec_input = torch.zeros(
            (batch_size, self.pred_len, 1),
            device=x.device
        )
        dec_input = torch.cat([feature_norm[:, -self.label_len:, :], dec_input], dim=1)
        
        # 미래 시간 특성 생성 (0으로 채움)
        future_time_features = torch.zeros(
            (batch_size, self.pred_len, time_features.shape[-1]), 
            device=x.device
        )
        time_features_dec = torch.cat(
            [time_features[:, -self.label_len:, :], future_time_features], 
            dim=1
        )
        
        # 인코더 통과
        enc_out, _ = self.encoder(enc_out)
        
        # 디코더 임베딩
        dec_out = self.dec_embedding(dec_input, time_features_dec)
        
        # 디코더 통과
        dec_out = self.decoder(dec_out, enc_out)
        
        # 예측 부분만 추출
        dec_out = dec_out[:, -self.pred_len:, :]
        
        # 역정규화
        dec_out = dec_out * stdev + mean
        
        return dec_out