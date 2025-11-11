import torch.nn as nn
from models.base_model import BaseModel
from layers.tcn import TemporalBlock

class TCNModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # TCN specific parameters
        self.num_channels = configs.get('num_channels', [32, 64, 128])
        self.kernel_size = configs.get('kernel_size', 3)
        self.dropout = configs.get('dropout', 0.2)
        
        # TCN layers 구성
        self.tcn_layers = nn.ModuleList()
        num_levels = len(self.num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            
            self.tcn_layers.append(
                TemporalBlock(
                    in_channels, out_channels, self.kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(self.kernel_size-1) * dilation_size,
                    dropout=self.dropout
                )
            )
            
        # 예측을 위한 추가 레이어
        self.pred_network = nn.Sequential(
            nn.Linear(self.num_channels[-1] * self.input_len, self.pred_len * self.output_size),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size = x.size(0)
        
        # Reshape for TCN: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # TCN layers
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
            
        # Reshape: [batch, channels * seq_len]
        x = x.reshape(batch_size, -1)
        
        # Generate predictions
        x = self.pred_network(x)
        
        # Reshape to expected output format: [batch, pred_len, features]
        x = x.reshape(batch_size, self.pred_len, self.output_size)
        
        return x