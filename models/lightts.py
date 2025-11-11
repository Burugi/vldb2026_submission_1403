import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from layers.lightts_layers import IEBlock

class LightTSModel(BaseModel):
    def __init__(self, configs):
        super().__init__(configs)
        
        # 모델 파라미터 설정
        self.d_model = configs.get('d_model', 512)
        self.chunk_size = configs.get('chunk_size', 24)
        
        # CD/CI 모드에 따른 입출력 설정
        if self.mode == 'CD':
            self.enc_in = self.output_size  # 출력 특성 수
        else:  # CI 모드
            self.enc_in = 1  # 단일 특성
            
        # 청크 크기 계산 및 패딩 조정
        self.chunk_size = min(self.pred_len, self.input_len, self.chunk_size)
        
        # 입력 길이가 chunk_size의 배수가 되도록 패딩 처리
        if self.input_len % self.chunk_size != 0:
            self.padded_input_len = self.input_len + (self.chunk_size - self.input_len % self.chunk_size)
        else:
            self.padded_input_len = self.input_len
            
        self.num_chunks = self.padded_input_len // self.chunk_size
        
        # 모델 레이어 구성
        self._build()
        
    def _build(self):
        # 연속적 샘플링을 위한 IEBlock
        self.layer_1 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )
        
        self.chunk_proj_1 = nn.Linear(self.num_chunks, 1)
        
        # 간헐적 샘플링을 위한 IEBlock
        self.layer_2 = IEBlock(
            input_dim=self.chunk_size,
            hid_dim=self.d_model // 4,
            output_dim=self.d_model // 4,
            num_node=self.num_chunks
        )
        
        self.chunk_proj_2 = nn.Linear(self.num_chunks, 1)
        
        # 특성 처리를 위한 IEBlock
        self.layer_3 = IEBlock(
            input_dim=self.d_model // 2,
            hid_dim=self.d_model // 2,
            output_dim=self.pred_len,
            num_node=self.enc_in
        )
        
        # AR 컴포넌트
        self.ar = nn.Linear(self.padded_input_len, self.pred_len)
        
    def forward(self, x):
        if self.mode == 'CD':
            return self._forward_cd(x)
        else:  # CI mode
            return self._forward_ci(x)
    
    def _forward_cd(self, x):
        # 데이터와 시간 특성 분리
        features = x[:, :, :self.output_size]
        time_features = x[:, :, self.output_size:]  # 안 사용됨
        
        # 정규화
        means = features.mean(1, keepdim=True).detach()
        x_enc = features - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # 입력 길이가 padded_input_len과 다르면 패딩 추가
        if x_enc.shape[1] < self.padded_input_len:
            padding = torch.zeros((x_enc.shape[0], self.padded_input_len - x_enc.shape[1], x_enc.shape[2]), 
                                 device=x_enc.device)
            x_enc = torch.cat([x_enc, padding], dim=1)
        
        # 예측 수행
        dec_out = self._encoder(x_enc)
        
        # 역정규화
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out
    
    def _forward_ci(self, x):
        # 데이터와 시간 특성 분리
        feature = x[:, :, 0:1]
        time_features = x[:, :, 1:]  # 안 사용됨
        
        # 정규화
        mean = feature.mean(1, keepdim=True).detach()
        x_enc = feature - mean
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        
        # 입력 길이가 padded_input_len과 다르면 패딩 추가
        if x_enc.shape[1] < self.padded_input_len:
            padding = torch.zeros((x_enc.shape[0], self.padded_input_len - x_enc.shape[1], x_enc.shape[2]), 
                                 device=x_enc.device)
            x_enc = torch.cat([x_enc, padding], dim=1)
        
        # 예측 수행
        dec_out = self._encoder(x_enc)
        
        # 역정규화
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (mean[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out
    
    def _encoder(self, x):
        B, T, N = x.size()
        
        # AutoRegressive 성분
        highway = self.ar(x.permute(0, 2, 1))
        highway = highway.permute(0, 2, 1)
        
        # 연속적 샘플링
        x1 = x.reshape(B, self.num_chunks, self.chunk_size, N)
        x1 = x1.permute(0, 3, 2, 1)
        x1 = x1.reshape(-1, self.chunk_size, self.num_chunks)
        x1 = self.layer_1(x1)
        x1 = self.chunk_proj_1(x1).squeeze(dim=-1)
        
        # 간헐적 샘플링
        x2 = x.reshape(B, self.chunk_size, self.num_chunks, N)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2.reshape(-1, self.chunk_size, self.num_chunks)
        x2 = self.layer_2(x2)
        x2 = self.chunk_proj_2(x2).squeeze(dim=-1)
        
        # 특성 결합
        x3 = torch.cat([x1, x2], dim=-1)
        x3 = x3.reshape(B, N, -1)
        x3 = x3.permute(0, 2, 1)
        
        # 최종 예측
        out = self.layer_3(x3)
        
        # AR 성분과 결합
        out = out + highway
        
        return out