import torch.nn as nn
from models.base_model import BaseModel
from layers.transformer import PositionalEncoding

class TransformerModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # Transformer specific parameters
        self.d_model = configs.get('d_model', 128)
        self.nhead = configs.get('nhead', 8)
        self.num_encoder_layers = configs.get('num_encoder_layers', 3)
        self.dim_feedforward = configs.get('dim_feedforward', 256)
        self.dropout = configs.get('dropout', 0.1)
        
        # Input projection
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_encoder_layers
        )
        
        # 예측을 위한 추가 레이어
        self.pred_network = nn.Sequential(
            nn.Linear(self.d_model * self.input_len, self.pred_len * self.output_size),
            nn.Dropout(self.dropout)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Reshape encoded sequence for prediction
        encoded = encoded.reshape(batch_size, -1)
        
        # Generate predictions
        x = self.pred_network(encoded)
        
        # Reshape to expected output format
        x = x.reshape(batch_size, self.pred_len, self.output_size)
        
        return x