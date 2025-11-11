import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# FreTS에 필요한 주파수 영역 MLP 함수
def FreMLP(B, nd, dimension, x, r, i, rb, ib, sparsity_threshold):
    """
    주파수 영역 MLP 함수
    
    Args:
        B: 배치 크기
        nd: 두 번째 차원의 크기
        dimension: FFT가 적용된 차원의 크기
        x: 입력 텐서 (주파수 영역)
        r: 실수 파트 가중치
        i: 허수 파트 가중치
        rb: 실수 파트 바이어스
        ib: 허수 파트 바이어스
        sparsity_threshold: 희소성 임계값
    
    Returns:
        주파수 영역에서 변환된 출력 텐서
    """
    o1_real = F.relu(
        torch.einsum('bijd,dd->bijd', x.real, r) - 
        torch.einsum('bijd,dd->bijd', x.imag, i) + 
        rb
    )

    o1_imag = F.relu(
        torch.einsum('bijd,dd->bijd', x.imag, r) + 
        torch.einsum('bijd,dd->bijd', x.real, i) + 
        ib
    )

    y = torch.stack([o1_real, o1_imag], dim=-1)
    y = F.softshrink(y, lambd=sparsity_threshold)
    y = torch.view_as_complex(y)
    return y