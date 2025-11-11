from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from sklearn.preprocessing import StandardScaler
from utils.timefeature import TimeFeature

class DataLoader:
    def __init__(self, data_path: str, file_name: str, features: list, date_column: str = 'date'):
        """
        Args:
            data_path: 데이터 경로
            file_name: 파일 이름
            features: 사용할 feature 리스트
            date_column: 날짜 컬럼명
        """
        self.data_path = Path(data_path)
        self.file_name = file_name
        self.features = features
        self.date_column = date_column
        self.scalers = {feature: StandardScaler() for feature in features}
        
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """데이터 로드 및 시간 특성 생성"""
        df = pd.read_csv(self.data_path / self.file_name)
        time_features = TimeFeature.create_time_features(df[self.date_column])
        return df[self.features], time_features
    
    def prepare_ci_sequences(self, data: np.ndarray, time_features: np.ndarray, 
                           target_feature: str, input_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """CI 모드용 시퀀스 생성"""
        feature_idx = self.features.index(target_feature)
        feature_data = data[:, feature_idx:feature_idx+1]
        
        sequences = []
        targets = []
        
        for i in range(len(data) - input_len - pred_len + 1):
            input_sequence = np.concatenate([
                feature_data[i:i+input_len],
                time_features[i:i+input_len]
            ], axis=1)
            target_sequence = feature_data[i+input_len:i+input_len+pred_len]
            
            sequences.append(input_sequence)
            targets.append(target_sequence)
            
        return np.array(sequences), np.array(targets)
    
    def prepare_cd_sequences(self, data: np.ndarray, time_features: np.ndarray, 
                           input_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """CD 모드용 시퀀스 생성"""
        sequences = []
        targets = []
        
        for i in range(len(data) - input_len - pred_len + 1):
            input_sequence = np.concatenate([
                data[i:i+input_len],
                time_features[i:i+input_len]
            ], axis=1)
            target_sequence = data[i+input_len:i+input_len+pred_len]
            
            sequences.append(input_sequence)
            targets.append(target_sequence)
            
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, input_len: int, pred_len: int, train_ratio: float = 0.7, 
                    val_ratio: float = 0.2, mode: str = 'CD', target_feature: str = None) -> Dict:
        """데이터 준비"""
        data, time_features = self.load_data()
        scaled_data = np.zeros_like(data.values)
        
        for i, feature in enumerate(self.features):
            scaled_data[:, i] = self.scalers[feature].fit_transform(data[feature].values.reshape(-1, 1)).ravel()
        
        if mode == 'CI':
            if target_feature is None:
                raise ValueError("CI 모드에서는 target_feature가 필요합니다.")
            x, y = self.prepare_ci_sequences(scaled_data, time_features, target_feature, input_len, pred_len)
        else:  # CD mode
            x, y = self.prepare_cd_sequences(scaled_data, time_features, input_len, pred_len)
        
        n_samples = len(x)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        return {
            'train': (x[:train_size], y[:train_size]),
            'val': (x[train_size:train_size+val_size], y[train_size:train_size+val_size]),
            'test': (x[train_size+val_size:], y[train_size+val_size:]),
            'scalers': self.scalers,
            'features': self.features,
            'time_feature_size': time_features.shape[1]
        }