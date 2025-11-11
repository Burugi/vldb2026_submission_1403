import torch
import torch.nn as nn
import torch.nn.functional as F

class Splitting(nn.Module):
    """시계열 데이터를 짝수/홀수 인덱스로 분리"""
    def __init__(self):
        super().__init__()
        
    def even(self, x):
        return x[:, ::2, :]
        
    def odd(self, x):
        return x[:, 1::2, :]
        
    def forward(self, x):
        return self.even(x), self.odd(x)

class CausalConvBlock(nn.Module):
    """인과성을 보장하는 1D 컨볼루션 블록"""
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReplicationPad1d((kernel_size - 1, 0)),  # 인과성 보장을 위한 padding
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),  # 1x1 convolution
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.conv_block(x)

class SCIBlock(nn.Module):
    """SCINet의 기본 블록"""
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super().__init__()
        self.splitting = Splitting()
        
        # 4개의 CausalConvBlock 인스턴스 생성
        self.modules_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.modules_odd = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_even = CausalConvBlock(d_model, kernel_size, dropout)
        self.interactor_odd = CausalConvBlock(d_model, kernel_size, dropout)
        
    def forward(self, x):
        # 데이터 분리
        x_even, x_odd = self.splitting(x)
        
        # 시계열 -> 채널 변환
        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)
        
        # Interactive Learning
        x_even_temp = x_even.mul(torch.exp(self.modules_even(x_odd)))
        x_odd_temp = x_odd.mul(torch.exp(self.modules_odd(x_even)))
        
        x_even_update = x_even_temp + self.interactor_even(x_odd_temp)
        x_odd_update = x_odd_temp - self.interactor_odd(x_even_temp)
        
        # 채널 -> 시계열 변환
        return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1)

class SCINetTree(nn.Module):
    """SCINet의 재귀적 구조"""
    def __init__(self, d_model, current_level=3, kernel_size=5, dropout=0.0):
        super().__init__()
        self.current_level = current_level
        self.working_block = SCIBlock(d_model, kernel_size, dropout)
        
        if current_level != 0:
            self.scinet_tree_odd = SCINetTree(d_model, current_level-1, kernel_size, dropout)
            self.scinet_tree_even = SCINetTree(d_model, current_level-1, kernel_size, dropout)
            
    def zip_up_the_pants(self, even, odd):
        """짝수/홀수 시퀀스 결합"""
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)
        
        min_len = min(even.shape[0], odd.shape[0])
        zipped_data = []
        
        for i in range(min_len):
            zipped_data.append(even[i].unsqueeze(0))
            zipped_data.append(odd[i].unsqueeze(0))
            
        if even.shape[0] > odd.shape[0]:
            zipped_data.append(even[-1].unsqueeze(0))
            
        return torch.cat(zipped_data, 0).permute(1, 0, 2)
        
    def forward(self, x):
        # 홀수 길이 처리
        odd_flag = False
        if x.shape[1] % 2 == 1:
            odd_flag = True
            x = torch.cat([x, x[:, -1:, :]], dim=1)
            
        # SCIBlock 적용
        x_even_update, x_odd_update = self.working_block(x)
        
        if odd_flag:
            x_odd_update = x_odd_update[:, :-1]
            
        # 재귀적 처리
        if self.current_level == 0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(
                self.scinet_tree_even(x_even_update),
                self.scinet_tree_odd(x_odd_update)
            )