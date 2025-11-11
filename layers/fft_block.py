import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def fft_for_period(x, k=2):
    """주기성 분석을 위한 FFT 연산"""
    # FFT 실행
    xf = torch.fft.rfft(x, dim=1)
    
    # 진폭으로 주기 찾기
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # DC 성분 제거
    
    # 상위 k개 주파수 선택
    _, top_list = torch.topk(frequency_list, min(k, len(frequency_list)-1))
    top_list = top_list.detach().cpu().numpy()
    
    # 주기 계산 (0으로 나누는 것 방지)
    periods = []
    for idx in top_list:
        if idx == 0:
            periods.append(x.shape[1])
        else:
            periods.append(max(2, x.shape[1] // (idx + 1)))
            
    return np.array(periods), abs(xf).mean(-1)[:, top_list]