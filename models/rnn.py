import torch
import torch.nn as nn
from models.base_model import BaseModel

class RNNModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # RNN specific parameters
        self.hidden_size = configs.get('hidden_size', 128)
        self.num_layers = configs.get('num_layers', 2)
        self.dropout = configs.get('dropout', 0.1)
        self.rnn_type = configs.get('rnn_type', 'LSTM')
        
        # RNN Layer
        rnn_class = getattr(nn, self.rnn_type)
        self.rnn = rnn_class(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # 예측을 위한 추가 레이어
        self.pred_network = nn.Sequential(
            nn.Linear(self.hidden_size * self.input_len, self.pred_len * self.output_size),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # RNN forward
        rnn_out, _ = self.rnn(x)  # [batch, input_len, hidden_size]
        
        # 전체 시퀀스 정보 활용
        rnn_out = rnn_out.reshape(batch_size, -1)  # [batch, input_len * hidden_size]
        
        # 예측 생성
        x = self.pred_network(rnn_out)
        
        # 출력 형식에 맞게 reshape
        x = x.reshape(batch_size, self.pred_len, self.output_size)
        
        return x
