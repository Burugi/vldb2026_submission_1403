# Does Channel Dependency Matter for Multivariate Time Series Forecasting? A Systematic Analysis across Interaction Scope, Level, and Architecture

This repository contains the experimental code for the VLDB 2026 submission titled **"Does Channel Dependency Matter for Multivariate Time Series Forecasting? A Systematic Analysis across Interaction Scope, Level, and Architecture"**.

## 📖 Abstract

This work presents a comprehensive empirical study that systematically compares channel dependency modeling strategies for multivariate time series forecasting (MTSF). We evaluate **18 deep learning models** across **10 benchmark datasets** using both **Channel-Dependent (CD)** and **Channel-Independent (CI)** approaches. Our novel three-dimensional analysis framework examines models along interaction architecture, scope, and level to provide deeper insights into when and how channel dependency matters.

## 🏗️ Framework Architecture

Our three-dimensional analysis framework categorizes models along:

### 1. Interaction Architecture
- **MLP-based**: Linear, DLinear, TSMixer, TimeMixer, TiDE, LightTS
- **RNN-based**: RNN, SegRNN, DSSRNN, SSRNN
- **CNN-based**: TCN, TimesNet, SCINet, MICN
- **Transformer-based**: Transformer, Autoformer, Informer, Pyraformer

### 2. Interaction Scope
- **Global**: Process all temporal steps simultaneously
- **Local**: Focus on limited temporal windows
- **Hierarchical**: Capture multi-scale temporal patterns
- **Sparse**: Selectively focus on important temporal steps

### 3. Interaction Level
- **Direct**: Raw temporal processing
- **Decomposition**: Trend-seasonal decomposition
- **Spectral**: Frequency domain analysis

## 📊 Included Models

- **DLinear** (+ Linear)
- **TCN**
- **Transformer**
- **RNN**
- **Autoformer**
- **TimeMixer**
- **TSMixer**
- **SegRNN**
- **SCINet**
- **TimesNet**
- **TiDE**
- **LightTS**
- **Pyraformer**
- **Informer**
- **DSSRNN** (+ SSRNN)
- **MICN**

## 📁 Project Structure

```
.
├── data/                          # Data loading and preprocessing
│   └── dataloader.py             # Unified data loader for all datasets
├── models/                        # Model implementations
│   ├── base_model.py             # Base model class
│   ├── autoformer.py             # Autoformer implementation
│   ├── dlinear.py                # DLinear implementation
│   └── ...                       # Other model implementations
├── layers/                        # Model layer implementations
│   ├── AutoCorrelation.py        # AutoCorrelation mechanisms
│   ├── decomposition.py          # Time series decomposition
│   └── ...                       # Model-specific layers
├── optimization/                  # Hyperparameter optimization
│   └── hyperopt.py               # Optuna-based hyperparameter tuning
├── training/                      # Training utilities
│   └── trainer.py                # Model training and evaluation
├── utils/                         # Utility functions
│   ├── masking.py                # Attention masking
│   └── timefeature.py            # Time feature engineering
├── configs/                       # Configuration files
│   └── hyperopt/                 # Hyperparameter search spaces
├── run_hyperopt.py               # Hyperparameter optimization script
├── run_training.py               # Model training script
├── overall_results_mse_with_CD_CI.csv               # Overall accuracy results
├── overall_results_scalability_with_CD_CI.csv       # Overall scalability results
└── requirements.txt              # Python dependencies
```

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Burugi/vldb2026_submission_1403.git
cd vldb2026_submission_1403
```

2. **Create conda environment**
```bash
conda create -n mtsf python=3.8
conda activate mtsf
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📊 Dataset Setup

Download the benchmark datasets from: 

The framework supports 10 benchmark datasets:
- **Milano_6165**: Telecommunications data in the 6165 section (5 channels)
- **Weather**: Meteorological data (21 channels) 
- **Electricity**: Power consumption data (321 channels)
- **Traffic**: Traffic flow data (862 channels)
- **ETTh1/ETTm1/ETTh2/ETTm2**: Electricity Transformer Temperature (7 channels each)
- **Exchange Rate**: Currency exchange rates (8 channels)

Place datasets in the `dataset/`, `milano_other/` directory with the following structure:
```
dataset/
    ├── milano_6165.csv
    ├── weather.csv
    ├── electricity.csv
    ├── traffic.csv
    ├── ETTh1.csv
    ├── ETTm1.csv
    ├── ETTh2.csv
    ├── ETTm2.csv
    ├── exchange_rate.csv
    milano_other/
        ├── milano_410.csv
        ├── milano_410.csv
        └── ...
```

## 🚀 Quick Start

### 1. Hyperparameter Optimization

```bash
python run_hyperopt.py \
  --file_name milano_6165.csv \
  --models dlinear tcn transformer \
  --modes CD CI \
  --target_feature OT \
  --n_trials 10
```

- `--file_name`: Target dataset file
- `--models`: List of target models
- `--modes`: Experiment modes (`CD`, `CI`)
- `--target_feature`: Feature to predict in CI mode
- `--n_trials`: Number of Optuna optimization trials

### 2. Model Training

```bash
python run_training.py \
  --file_name Milano.csv \
  --models dlinear tcn transformer \
  --modes CD CI \
  --target_feature OT \
  --n_repeats 5 \
  --epochs 50
```

- `--file_name`: Target dataset file
- `--models`: List of target models
- `--modes`: Experiment modes (`CD`, `CI`)
- `--target_feature`: Feature to predict in CI mode
- `--n_repeats`: Number of repeated experiments
- `--epochs`: Maximum training epochs

## ⚙️ Hyperparameter Search Spaces

# ✅ Hyperparameter Search Spaces (Summary)

### **Common Hyperparameters**

| Parameter         | Search Space            |
| ----------------- | ----------------------- |
| **learning_rate** | 1e-4 → 1e-2 (log scale) |
| **weight_decay**  | 1e-6 → 1e-2 (log scale) |
| **batch_size**    | {32, 64, 128, 256}      |
| **dropout**       | 0.1 → 0.5               |

---

# ✅ Model-Specific Hyperparameters

| Model           | Hyperparameters (Search Space)                                                                                                                                                                           |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Linear**      | None                                                                                                                                                                                                     |
| **DLinear**     | kernel_size: 15 → 45 (step 2)                                                                                                                                                                            |
| **TCN**         | hidden_size: {64,128,256,512}<br>num_layers: 1 → 4<br>rnn_type: {LSTM, GRU}                                                                                                                              |
| **Transformer** | d_model: {64,128,256}<br>nhead: {4,8}<br>num_encoder_layers: 1 → 4<br>dim_feedforward: {256,512,1024}                                                                                                    |
| **RNN**         | hidden_size: {64,128,256,512}<br>num_layers: 1 → 4<br>rnn_type: {LSTM, GRU}                                                                                                                              |
| **Autoformer**  | d_model: {64,128,256,512}<br>n_heads: {4,8}<br>e_layers: 1 → 3<br>d_layers: 1 → 3<br>d_ff: {128,256,512,1024}<br>moving_avg: {13,25,37}<br>factor: {1,5}                                                 |
| **TimeMixer**   | d_model: {64,128,256}<br>down_sampling_layers: 1 → 3<br>down_sampling_window: {2,4}<br>num_layers: 1 → 4<br>decomp_method: {moving_avg, dft}<br>moving_avg: 13 → 37 (step 2)<br>top_k: {3,7}             |
| **TSMixer**     | d_model: {64,128,256,512}<br>e_layers: 1 → 4                                                                                                                                                             |
| **SegRNN**      | d_model: {64,128,256,512}<br>seg_len: {6,12,24}<br>num_layers: 1 → 3                                                                                                                                     |
| **SCINet**      | levels: {2,4}<br>kernel_size: {3,5,7}<br>stacks: {1,2}                                                                                                                                                   |
| **TimesNet**    | d_model: {64,128,256,512}<br>e_layers: 1 → 4<br>d_ff: {128,256,512,1024}<br>top_k: {1,5}<br>num_kernels: {4,8}                                                                                           |
| **TiDE**        | d_model: {64,128,256,512}<br>feature_encode_dim: {2,8}<br>e_layers: 1 → 4<br>d_layers: 1 → 4<br>d_ff: {128,256,512,1024}                                                                                 |
| **LightTS**     | d_model: {64,128,256,512}<br>chunk_size: 4 → 36                                                                                                                                                          |
| **Pyraformer**  | d_model: {64,128,256,512}<br>d_ff: {128,256,512,1024}<br>n_heads: {4,8,16}<br>e_layers: 1 → 3<br>window_size: {[2,2],[4,4],[8,4]}<br>inner_size: {3,7}                                                   |
| **Informer**    | d_model: {64,128,256,512}<br>n_heads: {4,8}<br>e_layers: 1 → 3<br>d_layers: 1 → 3<br>d_ff: {128,256,512,1024}<br>factor: {1,5}<br>label_len: 0 → 48<br>activation: {relu, gelu}<br>distil: {true, false} |
| **DSSRNN**      | hidden_size: {64,128,256,512}<br>num_layers: 1 → 4<br>kernel_size: 3 → 25                                                                                                                                |
| **SSRNN**       | hidden_size: {64,128,256,512}<br>num_layers: 1 → 4                                                                                                                                                       |
| **MICN**        | d_model: {64,128,256,512}<br>n_heads: {4,16}<br>d_layers: 1 → 3<br>conv_kernel: {[12,16],[8,12],[16,24]}                                                                                                 |


## 📈 Key Findings

Our comprehensive evaluation reveals three key insights:

1. **CI approaches generally outperform in the majority of evaluated cases across both accuracy and efficiency metrics, yet no single model consistently dominates across all datasets.**

2. **Performance differences are not solely determined by domain properties but are closely linked to intrinsic channel relationships, as datasets with strong inter-channel correlations and high mutual information favor CD approaches.**

3. **Relatively complex models such as Transformer-based or Spectral and Hierarchical models favor CD approaches, particularly for datasets exhibiting high autocorrelation (ACF) and dependency forgetting rates (DFR), where strong and short-range temporal dependencies exist.**
