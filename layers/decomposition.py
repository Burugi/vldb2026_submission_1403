import torch
import torch.nn as nn

class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

class DFTDecomposition(nn.Module):
    """
    Series decomposition using DFT
    """
    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # DFT를 사용한 시계열 분해
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[:, :, 0] = 0  # DC 성분 제거
        
        # top-k 주파수 선택
        top_k_freq, _ = torch.topk(freq, self.top_k, dim=-1)
        mask = freq >= top_k_freq[:, :, -1:]
        xf = xf * mask
        
        # 계절성 및 트렌드 컴포넌트 추출
        x_season = torch.fft.irfft(xf, n=x.size(-1))
        x_trend = x - x_season
        
        return x_season, x_trend

class MovingAverageDecomposition(nn.Module):
    """
    Series decomposition using moving average
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # padding 처리
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # 이동 평균 계산
        moving_mean = self.avg(x_padded.transpose(1, 2)).transpose(1, 2)
        residual = x - moving_mean
        
        return residual, moving_mean