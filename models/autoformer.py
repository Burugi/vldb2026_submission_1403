import torch
import torch.nn as nn
from models.base_model import BaseModel
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

class AutoformerModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # 모델 파라미터 설정
        self.d_model = configs.get('d_model', 512)
        self.n_heads = configs.get('n_heads', 8)
        self.e_layers = configs.get('e_layers', 2)
        self.d_layers = configs.get('d_layers', 1)
        self.d_ff = configs.get('d_ff', 2048)
        self.moving_avg = configs.get('moving_avg', 25)
        self.factor = configs.get('factor', 1)
        self.dropout = configs.get('dropout', 0.1)
        self.embed = configs.get('embed', 'timeF')
        self.activation = configs.get('activation', 'gelu')
        self.output_attention = configs.get('output_attention', False)
        self.time_feature_size = configs.get('time_feature_size', 4)
        
        # 입출력 크기 설정
        if self.mode == 'CD':
            self.enc_in = self.input_size - self.time_feature_size  # 실제 feature 수
            self.dec_in = self.output_size
            self.c_out = self.output_size
        else:  # CI mode
            self.enc_in = 1  # 단일 feature
            self.dec_in = 1
            self.c_out = 1
            
        # 시계열 분해
        self.decomp = series_decomp(self.moving_avg)
            
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            self.enc_in, self.d_model, self.embed, dropout=self.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(
            self.dec_in, self.d_model, self.embed, dropout=self.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for _ in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, self.factor, attention_dropout=self.dropout,
                                      output_attention=False),
                        self.d_model, self.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                ) for _ in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim + time_feature]
        if self.mode == 'CI':
            return self._forward_ci(x)
        else:
            return self._forward_cd(x)

    def _forward_cd(self, x):
        # 데이터와 시간 특성 분리
        batch_size = x.shape[0]
        features = x[..., :-self.time_feature_size]
        time_features = x[..., -self.time_feature_size:]
        
        # Decomposition
        seasonal_init, trend_init = self.decomp(features)
        
        # Encoder
        enc_out = self.enc_embedding(features, time_features)
        enc_out, _ = self.encoder(enc_out)
        
        # Decoder 초기화
        dec_init = torch.zeros((batch_size, self.pred_len, self.dec_in), 
                             device=x.device)
        dec_time_features = torch.zeros((batch_size, self.pred_len, self.time_feature_size), 
                                      device=x.device)
        
        # Trend 초기화
        trend = trend_init[:, -1:, :].repeat(1, self.pred_len, 1)
        
        # Decoder
        dec_out = self.dec_embedding(dec_init, dec_time_features)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend)
        
        # 최종 예측 = seasonal + trend
        predictions = seasonal_part + trend_part
        
        return predictions

    def _forward_ci(self, x):
        # 데이터와 시간 특성 분리
        batch_size = x.shape[0]
        features = x[..., :1]  # 단일 feature
        time_features = x[..., 1:]
        
        # Decomposition
        seasonal_init, trend_init = self.decomp(features)
        
        # Encoder
        enc_out = self.enc_embedding(features, time_features)
        enc_out, _ = self.encoder(enc_out)
        
        # Decoder 초기화
        dec_init = torch.zeros((batch_size, self.pred_len, 1), 
                             device=x.device)
        dec_time_features = torch.zeros((batch_size, self.pred_len, self.time_feature_size), 
                                      device=x.device)
        
        # Trend 초기화
        trend = trend_init[:, -1:, :].repeat(1, self.pred_len, 1)
        
        # Decoder
        dec_out = self.dec_embedding(dec_init, dec_time_features)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, trend=trend)
        
        # 최종 예측 = seasonal + trend
        predictions = seasonal_part + trend_part
        
        return predictions