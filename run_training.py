# run_training.py
import argparse
from pathlib import Path
import torch
import yaml
import logging
from datetime import datetime
import json
import numpy as np
import os

from data.dataloader import DataLoader
from models.linear import LinearModel
from models.dlinear import DLinearModel
from models.tcn import TCNModel
from models.transformer import TransformerModel
from models.rnn import RNNModel
from models.autoformer import AutoformerModel
from models.timemixer import TimeMixerModel
from models.timesnet import TimesNetModel
from models.scinet import SCINetModel
from models.segrnn import SegRNNModel
from models.tsmixer import TSMixerModel
from models.tide import TiDEModel
from models.lightts import LightTSModel
from models.pyraformer import PyraformerModel
from models.informer import InformerModel
from models.dssrnn import DSSRNNModel
from models.ssrnn import SSRNNModel
from models.micn import MICNModel

from training.trainer import ModelTrainer

def delete_all_best_model_files(model_save_root):
    for root, dirs, files in os.walk(model_save_root):
        for file in files:
            if file == 'best_model.pth':
                file_path = os.path.join(root, file)
                os.remove(file_path)

# 로깅 설정
def setup_logging(log_dir='logs'):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train time series forecasting models')
    parser.add_argument('--data_path', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--file_name', type=str, default='Milano.csv',
                        help='Dataset file name')
    parser.add_argument('--features', type=str, nargs='+', 
                        default=["smsin", "smsout", "callin", "callout", "OT"],
                        help='Features to use')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['dlinear','tcn', 'transformer', 'rnn','timemixer','tsmixer','autoformer','segrnn','timesnet','scinet','tide', 'lightts', 'pyraformer', 'informer','dssrnn','ssrnn','micn','linear'], 
                        help='Models to train')
    parser.add_argument('--modes', type=str, nargs='+', default=['CI','CD'],
                        help='Forecasting modes (CD: Channel-Dependent, CI: Channel-Independent)')
    parser.add_argument('--target_feature', type=str, default='OT',
                        help='Target feature for CI mode')
    parser.add_argument('--hyperopt_dir', type=str, default='hyperopt_results_025',
                        help='Directory containing hyperparameter optimization results')
    parser.add_argument('--training_dir', type=str, default='training_results_025',
                        help='Directory to save training results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation')
    parser.add_argument('--n_repeats', type=int, default=5,
                        help='Number of repeated experiments')
    args = parser.parse_args()
    
    # 로깅 설정
    logger = setup_logging()
    logger.info(f"Arguments: {args}")
    
    # 데이터 로더 초기화
    loader = DataLoader(args.data_path, args.file_name, args.features)
    
    # 모델 매핑
    models = {
        'linear': LinearModel,
        'dlinear': DLinearModel,
        'tcn': TCNModel,
        'transformer': TransformerModel,
        'rnn': RNNModel,
        'autoformer': AutoformerModel,
        'timemixer': TimeMixerModel,
        'tsmixer': TSMixerModel,
        'segrnn': SegRNNModel,
        'scinet': SCINetModel,
        'timesnet': TimesNetModel,
        'tide': TiDEModel,
        'lightts': LightTSModel,
        'pyraformer': PyraformerModel,
        'informer': InformerModel,
        'dssrnn': DSSRNNModel,
        'ssrnn': SSRNNModel,
        'micn': MICNModel
    }
    
    # 선택된 모델만 처리
    selected_models = {k: v for k, v in models.items() if k in args.models}
    
    # 모델 학습 실행
    for model_name, model_class in selected_models.items():
        logger.info(f"Starting training for model: {model_name}")
        
        for mode in args.modes:
            if mode == 'CD':
                # CD 모드에서는 모든 특성을 함께 예측
                logger.info(f"Training {model_name} in CD mode")
                
                # 반복 실험 결과를 저장할 리스트
                all_histories = []
                
                for repeat in range(args.n_repeats):
                    logger.info(f"Repeat {repeat+1}/{args.n_repeats}")
                    
                    # 여기서 반복 실험을 위한 디렉토리를 미리 생성합니다
                    repeat_save_dir = Path(args.training_dir) / model_name / mode / f'repeat_{repeat+1}'
                    repeat_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    trainer = ModelTrainer(
                        model_class=model_class,
                        data_loader=loader,
                        model_name=model_name,
                        mode=mode,
                        hyperopt_dir=args.hyperopt_dir,
                        save_dir=str(repeat_save_dir.parent),  # 상위 디렉토리만 지정
                        device=args.device
                    )
                    
                    # save_dir을 repeat 폴더로 직접 지정
                    trainer.save_dir = repeat_save_dir
                    
                    # 모델 학습
                    history = trainer.train(epochs=args.epochs, patience=args.patience)
                    all_histories.append(history)
                    
                    # 개별 실험 결과 출력
                    logger.info(f"Repeat {repeat+1} - {model_name} - CD Mode Results:")
                    logger.info(f"Overall metrics: {history['test_metrics']['overall']}")
                    logger.info("Feature-wise metrics:")
                    for feature, metrics in history['test_metrics']['feature_wise'].items():
                        logger.info(f"{feature}: {metrics}")
                
            else:  # CI 모드
                # CI 모드에서는 지정된 특성만 예측
                logger.info(f"Training {model_name} in CI mode for feature {args.target_feature}")
                
                # 반복 실험 결과를 저장할 리스트
                all_histories = []
                
                for repeat in range(args.n_repeats):
                    logger.info(f"Repeat {repeat+1}/{args.n_repeats}")
                    
                    # 여기서 반복 실험을 위한 디렉토리를 미리 생성합니다
                    repeat_save_dir = Path(args.training_dir) / model_name / mode / args.target_feature / f'repeat_{repeat+1}'
                    repeat_save_dir.mkdir(parents=True, exist_ok=True)
                    
                    trainer = ModelTrainer(
                        model_class=model_class,
                        data_loader=loader,
                        model_name=model_name,
                        mode=mode,
                        target_feature=args.target_feature,
                        hyperopt_dir=args.hyperopt_dir, 
                        save_dir=str(repeat_save_dir.parent.parent),  # 상위 2단계 디렉토리만 지정
                        device=args.device
                    )
                    
                    # save_dir을 repeat 폴더로 직접 지정
                    trainer.save_dir = repeat_save_dir
                    
                    # 모델 학습
                    history = trainer.train(epochs=args.epochs, patience=args.patience)
                    all_histories.append(history)
                    
                    # 개별 실험 결과 출력
                    logger.info(f"Repeat {repeat+1} - {model_name} - CI Mode Results for {args.target_feature}:")
                    logger.info(f"Metrics: {history['test_metrics']}")
        
        model_save_root = Path(args.training_dir) / model_name
        delete_all_best_model_files(model_save_root)
                
if __name__ == '__main__':
    main()