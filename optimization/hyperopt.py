import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Type, Any
import yaml

class HyperOptimizer:
    def __init__(
        self,
        model_class: Type[torch.nn.Module],
        data_loader,
        model_name: str,
        mode: str = 'CD',
        target_feature: str = None,
        study_name: str = None,
        n_trials: int = 100,
        config_path: str = 'configs/hyperopt',
        save_dir: str = 'hyperopt_results_075',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model_class = model_class
        self.data_loader = data_loader
        self.model_name = model_name
        self.mode = mode
        self.target_feature = target_feature
        self.n_trials = n_trials
        self.device = device
        
        # CI 모드 검증
        if mode == 'CI' and target_feature is None:
            raise ValueError("CI 모드에서는 target_feature가 필요합니다.")

        # 하이퍼파라미터 설정 로드
        config_file = Path(config_path) / f'{model_name}.yaml'
        with open(config_file, 'r') as f:
            self.hyperopt_config = yaml.safe_load(f)

        # Study name 설정 수정
        if study_name is None:
            self.study_name = f"{model_name}_{mode}"
            if mode == 'CI':
                self.study_name += f"_{target_feature}"
            self.study_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.study_name = study_name
        
        # 결과 저장 디렉토리 설정
        self.save_dir = Path(save_dir) / self.study_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Study 생성
        self.study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=f"sqlite:///{self.save_dir}/study.db",
            load_if_exists=True
        )
        
    def _suggest_parameters(self, trial: Trial) -> dict:
        """모델별 하이퍼파라미터 제안"""
        params = {}
        
        # 공통 파라미터
        for param_name, param_config in self.hyperopt_config['common_params'].items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config['min'], 
                    param_config['max'], 
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max']
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )

        # 모델별 특정 파라미터
        if 'model_params' in self.hyperopt_config:
            for param_name, param_config in self.hyperopt_config['model_params'].items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config['min'], 
                        param_config['max'], 
                        log=param_config.get('log', False)
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
                    
        return params

    def train_and_validate(self, model, train_loader, val_loader, params):
        """모델 학습 및 검증"""
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = self.hyperopt_config.get('patience', 10)
        patience_counter = 0
        max_epochs = self.hyperopt_config.get('max_epochs', 50)
        
        for epoch in range(max_epochs):
            # 학습
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            
            # 검증
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = model(x)
                    val_loss = criterion(output, y)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        return best_val_loss

    def objective(self, trial: Trial) -> float:
        """최적화 목적 함수"""
        params = self._suggest_parameters(trial)
        
        # 데이터 준비
        data_dict = self.data_loader.prepare_data(
            input_len=96, 
            pred_len=96, 
            mode=self.mode,
            target_feature=self.target_feature
        )
        
        # 데이터 로더 생성
        train_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(data_dict['train'][0]),
            torch.FloatTensor(data_dict['train'][1])
        )
        val_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(data_dict['val'][0]),
            torch.FloatTensor(data_dict['val'][1])
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=8 # num_work 증가
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=params['batch_size'],
            num_workers=8
        )
        
        # 모델 초기화
        model_configs = {
            'input_len': 96,
            'pred_len': 96,
            'mode': self.mode,
            'num_features': len(self.data_loader.features),
            'time_feature_size': 4,
            **params
        }
        
        model = self.model_class(model_configs).to(self.device)
        
        # 학습 및 검증
        try:
            val_loss = self.train_and_validate(model, train_loader, val_loader, params)
            trial.set_user_attr('params', params)
            return val_loss
            
        except Exception as e:
            raise optuna.exceptions.TrialPruned()

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

    def optimize(self) -> Dict:
        """하이퍼파라미터 최적화 수행"""
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        # 최적의 하이퍼파라미터 저장
        best_params = self._convert_to_serializable(self.study.best_params)
        best_value = float(self.study.best_value)  # numpy.float32를 Python float로 변환
        
        results = {
            'model_name': self.model_name,
            'mode': self.mode,
            'target_feature': self.target_feature,
            'best_params': best_params,
            'best_value': best_value,
            'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        }
        
        # JSON 파일로 저장
        with open(self.save_dir / 'best_params.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        return results