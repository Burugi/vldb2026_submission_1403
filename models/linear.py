import torch
import torch.nn as nn
from models.base_model import BaseModel

class LinearModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # Simple Linear layers for direct time series prediction
        self.Linear = nn.Linear(self.input_len, self.pred_len)
        
        # 출력 조정을 위한 Linear
        if self.mode == 'CD':
            self.projection = nn.Linear(self.input_size, self.output_size)
        else:  # CI
            self.projection = nn.Linear(self.input_size, 1)
        
    def forward(self, x):
        # Direct linear transformation on time dimension
        # x shape: [batch, input_len, features]
        
        # Apply linear transformation: input_len → pred_len
        x = self.Linear(x.transpose(1,2)).transpose(1,2)
        # [batch, input_len, features] → [batch, features, input_len] 
        # → [batch, features, pred_len] → [batch, pred_len, features]
        
        # Project to the desired output size
        x = self.projection(x)
        
        return x