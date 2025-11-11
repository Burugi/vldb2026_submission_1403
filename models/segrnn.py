import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.segrnn_layers import ValueEmbedding, RNNEncoder, PositionalChannelEmbedding

class SegRNNModel(BaseModel):
    def __init__(self, configs):
        """
        Args:
            configs: 모델 설정값을 담은 딕셔너리
                - input_len: 입력 시퀀스 길이
                - pred_len: 예측 시퀀스 길이
                - mode: 'CD' 또는 'CI'
                - num_features: 특성 수
                - time_feature_size: 시간 특성 차원
                - d_model: 모델의 hidden dimension
                - seg_len: 세그먼트 길이
                - dropout: dropout 비율
        """
        super().__init__(configs)
        
        # 시계열 데이터와 시간 특성 분리
        self.data_feature_size = configs['num_features'] if self.mode == 'CD' else 1
        
        # 모델 파라미터
        self.d_model = configs.get('d_model', 512)
        self.seg_len = configs.get('seg_len', 12)
        self.dropout = configs.get('dropout', 0.1)
        
        # 세그먼트 수 계산
        self.seg_num_x = self.input_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len
        
        # Value Embedding
        self.value_embedding = ValueEmbedding(
            seg_len=self.seg_len,
            d_model=self.d_model
        )
        
        # RNN Encoder
        self.rnn = RNNEncoder(
            input_size=self.d_model,
            hidden_size=self.d_model,
            dropout=self.dropout
        )
        
        # Positional and Channel Embedding
        self.pos_channel_emb = PositionalChannelEmbedding(
            seg_num_y=self.seg_num_y,
            enc_in=self.data_feature_size,
            d_model=self.d_model
        )
        
        # Prediction Layer
        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
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
        
        # 마지막 값 저장 (정규화 용도)
        last_value = x_data[:, -1:, :].detach()
        
        # 정규화 (차분)
        x_data = x_data - last_value
        
        # 차원 변환: [batch, seq_len, feature] -> [batch, feature, seq_len]
        x_data = x_data.permute(0, 2, 1)
        
        # 세그먼트로 분할 및 임베딩
        # [batch * feature, seg_num_x, seg_len] -> [batch * feature, seg_num_x, d_model]
        x_segments = x_data.reshape(-1, self.seg_num_x, self.seg_len)
        x_embedded = self.value_embedding(x_segments)
        
        # RNN 인코딩
        _, h_n = self.rnn(x_embedded)
        
        # 위치 및 채널 임베딩 생성 및 RNN 디코딩
        pos_channel_emb = self.pos_channel_emb(batch_size)
        _, h_y = self.rnn(pos_channel_emb, h_n.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
        
        # 예측값 생성
        y = self.predict(h_y)
        y = y.view(batch_size, self.data_feature_size, self.pred_len)
        
        # 차원 복원 및 정규화 복원
        y = y.permute(0, 2, 1) + last_value
        
        return y

    def configure_optimizers(self):
        """옵티마이저 설정"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)