# run_hyperopt.py
import argparse
from pathlib import Path
import torch
import yaml
import logging
from datetime import datetime

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

from optimization.hyperopt import HyperOptimizer

# 로깅 설정
def setup_logging(log_dir='logs'):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter optimization for time series models')
    parser.add_argument('--config', type=str, default='configs/hyperopt_config.yaml', 
                        help='Configuration file path')
    parser.add_argument('--data_path', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--file_name', type=str, default='Milano.csv',
                        help='Dataset file name')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['dlinear', 'tcn', 'transformer', 'rnn','autoformer','timemixer','tsmixer','segrnn','scinet','timesnet','tide', 'lightts', 'pyraformer', 'informer','dssrnn','ssrnn','micn','linear'], 
                        help='Models to optimize')
    parser.add_argument('--modes', type=str, nargs='+', default=['CI','CD'], 
                        help='Forecasting modes (CD: Channel-Dependent, CI: Channel-Independent)')
    parser.add_argument('--target_feature', type=str, default='OT',
                        help='Target feature for CI mode')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of optimization trials')
    parser.add_argument('--device', type=str, default='cuda', #  if torch.cuda.is_available() else 'cpu'
                        help='Device to use for computation')
    args = parser.parse_args()

    # 로깅 설정
    logger = setup_logging()
    logger.info(f"Arguments: {args}")
    
    # 구성 로드
    config = load_config(args.config)
    
    # 데이터 설정
    FEATURES = config.get('features', ["smsin", "smsout", "callin", "callout", "OT"])
    
    # 데이터 로더 초기화
    loader = DataLoader(args.data_path, args.file_name, FEATURES)
    
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
    
    # 하이퍼파라미터 최적화 실행
    for model_name, model_class in selected_models.items():
        logger.info(f"Starting optimization for model: {model_name}")
        
        for mode in args.modes:
            if mode == 'CD':
                # CD 모드에서는 모든 특성을 함께 예측
                logger.info(f"Optimizing {model_name} in CD mode")
                optimizer = HyperOptimizer(
                    model_class=model_class,
                    data_loader=loader,
                    model_name=model_name,
                    mode=mode,
                    n_trials=args.n_trials,
                    device=args.device
                )
                
                results = optimizer.optimize()
                logger.info(f"{model_name} - CD Mode Results:")
                logger.info(f"Best parameters: {results['best_params']}")
                logger.info(f"Best value: {results['best_value']:.4f}")
                
            elif mode == 'CI':
                # CI 모드에서는 지정된 특성만 예측
                logger.info(f"Optimizing {model_name} in CI mode for feature {args.target_feature}")
                optimizer = HyperOptimizer(
                    model_class=model_class,
                    data_loader=loader,
                    model_name=model_name,
                    mode=mode,
                    target_feature=args.target_feature,
                    n_trials=args.n_trials,
                    device=args.device
                )
                
                results = optimizer.optimize()
                logger.info(f"{model_name} - CI Mode Results for {args.target_feature}:")
                logger.info(f"Best parameters: {results['best_params']}")
                logger.info(f"Best value: {results['best_value']:.4f}")

if __name__ == '__main__':
    main()