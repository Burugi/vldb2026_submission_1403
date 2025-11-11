import torch
import torch.nn as nn
import torch.nn.functional as F

class MovingAverage(nn.Module):
    """Moving average block for seasonal-trend decomposition"""
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x

class SeriesDecomp(nn.Module):
    """Series decomposition block"""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean

class AutoCorrelation(nn.Module):
    """AutoCorrelation module for time series"""
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        
    def time_delay_agg(self, values, corr):
        """Time delay aggregation"""
        batch, head, channel, length = values.shape
        
        # Top k autocorrelation selection
        top_k = int(self.factor * length)
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delays = torch.topk(mean_value, top_k, dim=-1)
        
        # Aggregation
        tmp_corr = torch.softmax(weights, dim=-1)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(values, -int(delays[0, i]), -1)
            delays_agg = delays_agg + pattern * (tmp_corr[:, i].view(-1, 1, 1, 1))
        return delays_agg

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # Period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # Time delay aggregation
        V = self.time_delay_agg(values.permute(0, 2, 3, 1).contiguous(), corr)
        V = V.permute(0, 3, 1, 2)
        
        return V.contiguous()