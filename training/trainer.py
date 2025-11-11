import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelTrainer:
    def __init__(
        self,
        model_class,
        data_loader,
        model_name: str,
        mode: str = 'CD',
        target_feature: str = None,
        hyperopt_dir: str = None,
        save_dir: str = 'training_results',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model_class = model_class
        self.data_loader = data_loader
        self.model_name = model_name
        self.mode = mode
        self.target_feature = target_feature
        self.device = device
        
        # 하이퍼파라미터 로드 부분 수정
        if hyperopt_dir:
            # 가장 최근의 하이퍼파라미터 최적화 결과를 찾음
            hyperopt_base = Path(hyperopt_dir)
            pattern = f"{model_name}_{mode}"
            if mode == 'CI':
                pattern += f"_{target_feature}"
            pattern += "_*"
            
            # 해당 패턴의 모든 디렉토리를 찾아서 가장 최근 것을 선택
            matching_dirs = sorted(hyperopt_base.glob(pattern), reverse=True)
            if not matching_dirs:
                print(f"Warning: No hyperopt results found for {pattern}. Using default parameters.")
                self.params = self._get_default_params()
            else:
                latest_dir = matching_dirs[0]
                with open(latest_dir / 'best_params.json', 'r') as f:
                    hyperopt_results = json.load(f)
                    self.params = hyperopt_results['best_params']
                    print(f"Loaded hyperparameters from: {latest_dir}")
        else:
            self.params = self._get_default_params()
        
        # 결과 저장 디렉토리 설정
        self.save_dir = Path(save_dir) / model_name / mode
        if mode == 'CI':
            self.save_dir = self.save_dir / target_feature
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # History 초기화
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': None
        }
    
    def _get_default_params(self) -> Dict[str, Any]:
        """기본 하이퍼파라미터 설정"""
        return {
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 32,
            'dropout': 0.1
        }
    
    def create_data_loaders(self, data_dict):
        """데이터 로더 생성"""
        train_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(data_dict['train'][0]),
            torch.FloatTensor(data_dict['train'][1])
        )
        val_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(data_dict['val'][0]),
            torch.FloatTensor(data_dict['val'][1])
        )
        test_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(data_dict['test'][0]),
            torch.FloatTensor(data_dict['test'][1])
        )
        
        batch_size = self.params.get('batch_size', 32)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,num_workers=8)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=8)
        
        return train_loader, val_loader, test_loader

    def train(self, epochs: int = 100, patience: int = 10):
        """모델 학습"""
        # 데이터 준비
        data_dict = self.data_loader.prepare_data(
            input_len=96,
            pred_len=96,
            mode=self.mode,
            target_feature=self.target_feature
        )
        train_loader, val_loader, test_loader = self.create_data_loaders(data_dict)
        
        # 모델 초기화
        model_configs = {
            'input_len': 96,
            'pred_len': 96,
            'mode': self.mode,
            'num_features': len(self.data_loader.features),
            'time_feature_size': 4,
            **self.params
        }
        model = self.model_class(model_configs).to(self.device)
        
        # Optimizer와 criterion 설정
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.params['learning_rate'],
            weight_decay=self.params['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # 학습
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = self.save_dir / 'best_model.pth'
        
        for epoch in range(epochs):
            # 학습
            model.train()
            train_losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # 검증
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    val_loss = criterion(output, y)
                    val_losses.append(val_loss.item())
            
            # Loss 기록
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 최적 모델 로드
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        
        # 테스트 데이터로 평가
        test_metrics = self.evaluate(model, test_loader)
        self.history['test_metrics'] = test_metrics
        
        # 결과 저장
        self.save_results()
        
        return self.history

    def evaluate(self, model, test_loader):
        """테스트 데이터로 모델 평가"""
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                predictions.append(output.cpu().numpy())
                actuals.append(y.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        # 3차원 배열을 2차원으로 변환
        pred_2d = predictions.reshape(-1, predictions.shape[-1])
        true_2d = actuals.reshape(-1, actuals.shape[-1])
        
        # 메트릭 계산
        if self.mode == 'CI':
            metrics = {
                'mse': mean_squared_error(true_2d, pred_2d),
                'rmse': np.sqrt(mean_squared_error(true_2d, pred_2d)),
                'mae': mean_absolute_error(true_2d, pred_2d)
            }
        else:
            metrics = {
                'overall': {
                    'mse': mean_squared_error(true_2d, pred_2d),
                    'rmse': np.sqrt(mean_squared_error(true_2d, pred_2d)),
                    'mae': mean_absolute_error(true_2d, pred_2d)
                },
                'feature_wise': {}
            }
            
            for i, feature in enumerate(self.data_loader.features):
                metrics['feature_wise'][feature] = {
                    'mse': mean_squared_error(true_2d[:, i], pred_2d[:, i]),
                    'rmse': np.sqrt(mean_squared_error(true_2d[:, i], pred_2d[:, i])),
                    'mae': mean_absolute_error(true_2d[:, i], pred_2d[:, i])
                }
        
        return metrics

    def _convert_to_serializable(self, obj):
        """NumPy/Torch 타입을 Python 기본 타입으로 변환"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
        
    def get_predictions(self, data_split='test'):
        """지정된 데이터 분할에 대한 예측 수행"""
        # 데이터 준비
        data_dict = self.data_loader.prepare_data(
            input_len=96,
            pred_len=96,
            mode=self.mode,
            target_feature=self.target_feature
        )
        
        # 데이터 로더 생성
        if data_split == 'test':
            data = torch.utils.data.TensorDataset(
                torch.FloatTensor(data_dict['test'][0]),
                torch.FloatTensor(data_dict['test'][1])
            )
        elif data_split == 'val':
            data = torch.utils.data.TensorDataset(
                torch.FloatTensor(data_dict['val'][0]),
                torch.FloatTensor(data_dict['val'][1])
            )
        else:
            raise ValueError(f"Invalid data split: {data_split}")
            
        data_loader = torch.utils.data.DataLoader(
            data, 
            batch_size=self.params.get('batch_size', 32),
            num_workers=8
        )

        # 모델 초기화 및 최적 가중치 로드
        model_configs = {
            'input_len': 96,
            'pred_len': 96,
            'mode': self.mode,
            'num_features': len(self.data_loader.features),
            'time_feature_size': 4,
            **self.params
        }
        model = self.model_class(model_configs).to(self.device)
        
        best_model_path = self.save_dir / 'best_model.pth'
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        else:
            raise ValueError("No trained model found")

        # 예측 수행
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = model(x)
                predictions.append(output.cpu().numpy())
                actuals.append(y.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        
        # 역스케일링 수행
        if self.mode == 'CD':
            predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])
            actuals_reshaped = actuals.reshape(-1, actuals.shape[-1])
            
            predictions_unscaled = np.zeros_like(predictions_reshaped)
            actuals_unscaled = np.zeros_like(actuals_reshaped)
            
            for i, feature in enumerate(self.data_loader.features):
                predictions_unscaled[:, i] = self.data_loader.scalers[feature].inverse_transform(
                    predictions_reshaped[:, i].reshape(-1, 1)
                ).ravel()
                actuals_unscaled[:, i] = self.data_loader.scalers[feature].inverse_transform(
                    actuals_reshaped[:, i].reshape(-1, 1)
                ).ravel()
                
            predictions = predictions_unscaled.reshape(predictions.shape)
            actuals = actuals_unscaled.reshape(actuals.shape)
        else:  # CI 모드
            if self.target_feature:
                scaler = self.data_loader.scalers[self.target_feature]
                predictions = scaler.inverse_transform(
                    predictions.reshape(-1, 1)
                ).ravel()
                actuals = scaler.inverse_transform(
                    actuals.reshape(-1, 1)
                ).ravel()
        
        return actuals, predictions
    
    def save_results(self):
        """학습 결과 저장"""
        results = {
            'model_name': self.model_name,
            'mode': self.mode,
            'target_feature': self.target_feature,
            'params': self._convert_to_serializable(self.params),
            'history': {
                'train_loss': self._convert_to_serializable(self.history['train_loss']),
                'val_loss': self._convert_to_serializable(self.history['val_loss']),
                'test_metrics': self._convert_to_serializable(self.history['test_metrics'])
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        with open(self.save_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=4)