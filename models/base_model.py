import torch
import torch.nn as nn
from typing import Dict

class BaseModel(nn.Module):
    def __init__(self, configs: Dict):
        super().__init__()
        # 기본 설정
        self.input_len = configs.get('input_len', 36)
        self.pred_len = configs.get('pred_len', 24)
        self.mode = configs.get('mode', 'CD')
        
        # Feature 관련 설정
        if self.mode == 'CD':
            self.input_size = configs['num_features'] + configs['time_feature_size']
            self.output_size = configs['num_features']
        else:  # CI mode
            self.input_size = 1 + configs['time_feature_size']
            self.output_size = 1
            
        # 학습 관련 설정
        self.learning_rate = configs.get('learning_rate', 0.001)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, x):
        raise NotImplementedError
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)