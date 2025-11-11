import torch
import torch.nn as nn
from models.base_model import BaseModel
from layers.decomposition import DFTDecomposition, MovingAverageDecomposition
from layers.mixing import TimeMixing

class TimeMixerModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # 기본 설정
        self.seq_len = configs.get('input_len', 36)
        self.pred_len = configs.get('pred_len', 24)
        self.down_sampling_layers = configs.get('down_sampling_layers', 2)
        self.down_sampling_window = configs.get('down_sampling_window', 2)
        self.d_model = configs.get('d_model', 128)
        self.dropout = configs.get('dropout', 0.1)
        self.num_layers = configs.get('num_layers', 3)
        
        # 분해 방식 설정
        decomp_method = configs.get('decomp_method', 'moving_avg')
        if decomp_method == 'dft':
            self.decomposition = DFTDecomposition(top_k=configs.get('top_k', 5))
        else:
            self.decomposition = MovingAverageDecomposition(kernel_size=configs.get('moving_avg', 25))
            
        # TimeMixer 레이어 스택
        self.time_mixer_layers = nn.ModuleList([
            TimeMixing(
                seq_len=self.seq_len,
                d_model=self.d_model,
                down_sampling_layers=self.down_sampling_layers,
                down_sampling_window=self.down_sampling_window
            ) for _ in range(self.num_layers)
        ])
        
        # 입력 프로젝션
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        
        # 다운샘플링 레이어
        self.down_sampling = nn.ModuleList([
            nn.AvgPool1d(
                kernel_size=self.down_sampling_window,
                stride=self.down_sampling_window
            ) for _ in range(self.down_sampling_layers)
        ])
        
        # 예측 레이어
        self.predict_layers = nn.ModuleList([
            nn.Linear(
                self.seq_len // (self.down_sampling_window ** i),
                self.pred_len
            ) for i in range(self.down_sampling_layers + 1)
        ])
        
        # 출력 프로젝션
        self.output_projection = nn.Linear(self.d_model, self.output_size)
        
    def _create_multi_scale_data(self, x):
        """다중 스케일 데이터 생성"""
        x_list = [x]
        x_scaled = x.transpose(1, 2)  # [batch, seq_len, feature] -> [batch, feature, seq_len]
        
        for down_sample in self.down_sampling:
            x_scaled = down_sample(x_scaled)
            x_list.append(x_scaled.transpose(1, 2))
            
        return x_list
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 입력 프로젝션
        x = self.input_projection(x)
        
        # 다중 스케일 데이터 생성
        x_list = self._create_multi_scale_data(x)
        
        # TimeMixer 레이어 통과
        for mixer in self.time_mixer_layers:
            x_list = mixer(x_list, self.decomposition)
        
        # 각 스케일에서 예측 수행
        predictions = []
        for i, (x_scale, predict_layer) in enumerate(zip(x_list, self.predict_layers)):
            pred = predict_layer(x_scale.transpose(1, 2)).transpose(1, 2)
            predictions.append(pred)
            
        # 모든 스케일의 예측 결과 통합
        final_pred = torch.stack(predictions, dim=-1).mean(-1)
        
        # 출력 프로젝션
        final_pred = self.output_projection(final_pred)
        
        return final_pred