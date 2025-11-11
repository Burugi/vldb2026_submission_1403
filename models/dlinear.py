import torch
import torch.nn as nn
from models.base_model import BaseModel
from layers.decomposition import MovingAverage

class DLinearModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        self.decomposition = MovingAverage(configs.get('kernel_size', 25), stride=1)
        
        # Seasonal Part
        self.Linear_Seasonal = nn.Linear(self.input_len, self.pred_len)
        # Trend Part
        self.Linear_Trend = nn.Linear(self.input_len, self.pred_len)
        
        # 출력 조정을 위한 Linear
        if self.mode == 'CD':
            self.projection = nn.Linear(self.input_size, self.output_size)
        else:  # CI
            self.projection = nn.Linear(self.input_size, 1)
        
    def forward(self, x):
        # decomposition
        seasonal_init = x - self.decomposition(x)
        trend_init = self.decomposition(x)
        
        # Seasonal
        seasonal_output = self.Linear_Seasonal(seasonal_init.transpose(1,2)).transpose(1,2)
        # Trend
        trend_output = self.Linear_Trend(trend_init.transpose(1,2)).transpose(1,2)
        
        # Combine seasonal and trend
        x = seasonal_output + trend_output
        
        # Project to the desired output size
        x = self.projection(x)
        
        return x