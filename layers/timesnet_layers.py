import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math

class FFTBlock:
    """FFT를 이용한 주기성 분석"""
    @staticmethod
    def get_frequency_domains(x, k=2):
        """
        입력 시퀀스의 주요 주파수 도메인을 찾습니다.
        
        Args:
            x: 입력 시퀀스 [B, T, C]
            k: 상위 k개의 주파수를 선택
            
        Returns:
            periods: 주요 주기 리스트
            weights: 각 주기의 가중치
        """
        # FFT 변환
        xf = torch.fft.rfft(x, dim=1)
        
        # 주파수별 진폭 계산
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0  # DC 성분 제거
        
        # 상위 k개 주파수 선택
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        
        # 주기 계산
        periods = x.shape[1] // top_list
        
        return periods, abs(xf).mean(-1)[:, top_list]

class InceptionBlock(nn.Module):
    """다중 커널 사이즈를 가진 Inception 블록"""
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(
                in_channels, 
                out_channels,
                kernel_size=2 * i + 1,
                padding=i
            ) for i in range(num_kernels)
        ])
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # 각 커널 사이즈별 결과 계산
        res_list = [kernel(x) for kernel in self.kernels]
        # 결과 평균
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    """TimesNet의 기본 블록"""
    def __init__(self, input_size, hidden_size, top_k=2, num_kernels=6):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.top_k = top_k
        
        # Inception 블록
        self.conv = nn.Sequential(
            InceptionBlock(input_size, hidden_size, num_kernels),
            nn.GELU(),
            InceptionBlock(hidden_size, input_size, num_kernels)
        )
        
    def forward(self, x):
        B, T, N = x.size()
        
        # 주기성 분석
        period_list, period_weight = FFTBlock.get_frequency_domains(x, self.top_k)
        
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            
            # padding
            if T % period != 0:
                padding_size = (((T - 1) // period) + 1) * period - T
                padding = torch.zeros(B, padding_size, N).to(x.device)
                out = torch.cat([x, padding], dim=1)
                length = out.shape[1]
            else:
                length = T
                out = x
                
            # 2D reshape
            out = out.reshape(B, length // period, period, N)
            out = out.permute(0, 3, 1, 2)
            
            # Inception block 적용
            out = self.conv(out)
            
            # 원래 shape로 복원
            out = out.permute(0, 2, 3, 1)
            out = out.reshape(B, -1, N)
            res.append(out[:, :T, :])
            
        # 결과 집계
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)
        
        # 가중 평균
        res = torch.sum(res * period_weight, -1)
        
        return res + x  # residual connection