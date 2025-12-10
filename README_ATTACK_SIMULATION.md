# Federated Learning Attack Simulation with Flower

This project implements a comprehensive federated learning attack simulation using the Flower framework and CIFAR-10 dataset. It evaluates the impact of label flipping attacks on model performance under different scenarios.

## Project Structure

```
flower2025/
├── flower2025/
│   ├── __init__.py
│   ├── client_app.py      # Client-side FL logic with attack support
│   ├── server_app.py      # Server-side FL logic (FedAvg & FedProx)
│   └── task.py            # Model, data loading, training, and evaluation
├── run_experiments.py     # Automated experiment runner
├── analyze_results.py     # Results analysis and visualization
├── plot.py                # Simple plotting utility
├── pyproject.toml         # Project configuration
└── results/               # Generated experiment results (created after running)
    ├── metrics_*.json     # Metrics for each experiment
    └── final_model_*.pt   # Trained models
```

## Features Implemented

### 1. Attack Mechanisms
- **Label Flipping Attack**: Random label flipping for malicious clients
- **Targeted Label Flipping**: Systematic label shifting (optional)
- Configurable attacker clients (0, 1, or 2 attackers)

### 2. Data Distributions
- **IID**: Independent and Identically Distributed data using `IidPartitioner`
- **Non-IID**: Heterogeneous data distribution using `DirichletPartitioner` (α=0.5)

### 3. Aggregation Strategies
- **FedAvg**: Standard Federated Averaging
- **FedProx**: Federated Proximal with regularization (μ=0.1)

### 4. Metrics Logged
For each communication round:
- Accuracy
- F1 Score (macro)
- Cohen's Kappa
- ROC-AUC (multi-class)
- Loss

### 5. Experimental Scenarios

**Part 2.1: FedAvg**
- IID Data: Baseline, 1 attacker, 2 attackers
- Non-IID Data: Baseline, 1 attacker, 2 attackers

**Part 2.2: FedProx**
- IID Data: Baseline, 1 attacker, 2 attackers
- Non-IID Data: Baseline, 1 attacker, 2 attackers

**Total: 12 experiments**

## Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Install Dependencies

```bash
# Activate your virtual environment
# On Windows PowerShell:
.\flower_env\Scripts\Activate.ps1

# On Linux/Mac:
source flower_env/bin/activate

# Install required packages
pip install flwr[simulation]>=1.23.0
pip install flwr-datasets[vision]>=0.5.0
pip install torch==2.7.1
pip install torchvision==0.22.1
pip install matplotlib
pip install scikit-learn
```

## Usage

### Option 1: Run All Experiments Automatically (Recommended)

```bash
# Run all 12 experiments
python run_experiments.py
```

This will:
- Run all combinations of strategies, data distributions, and attacker counts
- Save results in `results/` directory
- Take approximately 2-4 hours depending on hardware

### Option 2: Run Individual Experiments

```bash
# Baseline with FedAvg and IID data
flwr run . --run-config num-server-rounds=50 use-iid=true strategy=FedAvg attacker-ids=[]

# 1 attacker with FedAvg and IID data
flwr run . --run-config num-server-rounds=50 use-iid=true strategy=FedAvg attacker-ids=[0] attack-type=label_flipping

# 2 attackers with FedProx and non-IID data
flwr run . --run-config num-server-rounds=50 use-iid=false strategy=FedProx attacker-ids=[0,1] attack-type=label_flipping proximal-mu=0.1
```

### Analyze Results and Generate Plots

```bash
# After experiments complete, generate analysis and plots
python analyze_results.py
```

This will create:
- `plots/`: Directory with all comparison plots
  - Per-strategy comparisons (baseline vs attackers)
  - Cross-strategy comparisons (FedAvg vs FedProx)
  - Summary dashboard with final metrics
- `analysis_report.txt`: Detailed text report with final metrics

## Configuration

Edit `pyproject.toml` to adjust:

```toml
[tool.flwr.app.config]
num-server-rounds = 50      # Number of communication rounds
fraction-train = 1          # Fraction of clients selected per round
local-epochs = 1            # Local training epochs per round
lr = 0.01                   # Learning rate

[tool.flwr.federations.local-simulation]
options.num-supernodes = 5  # Number of clients
```

## Experimental Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rounds | 50 | Communication rounds |
| Clients | 5 | Total participating clients |
| Batch Size | 32 | Training batch size |
| Model | Custom CNN | 2 conv layers + 2 FC layers |
| Optimizer | AdamW | Weight decay: 1e-4 |
| Learning Rate | 0.001 | Fixed throughout training |
| Attack Type | Label Flipping | Random label assignment |
| IID Partitioner | IidPartitioner | Equal random distribution |
| Non-IID Partitioner | DirichletPartitioner | α=0.5 for heterogeneity |
| FedProx μ | 0.1 | Proximal regularization term |

## Results Structure

Each experiment generates:

```
results/
├── metrics_FedAvg_iid_0attackers.json          # Baseline
├── metrics_FedAvg_iid_1attackers.json          # 1 attacker
├── metrics_FedAvg_iid_2attackers.json          # 2 attackers
├── metrics_FedAvg_non_iid_0attackers.json
├── metrics_FedAvg_non_iid_1attackers.json
├── metrics_FedAvg_non_iid_2attackers.json
├── metrics_FedProx_iid_0attackers.json
├── metrics_FedProx_iid_1attackers.json
├── metrics_FedProx_iid_2attackers.json
├── metrics_FedProx_non_iid_0attackers.json
├── metrics_FedProx_non_iid_1attackers.json
└── metrics_FedProx_non_iid_2attackers.json
```

Each JSON file contains:
- `experiment_config`: Experiment parameters
- `train_metrics_clientapp`: Training metrics from clients
- `evaluate_metrics_clientapp`: Client evaluation metrics
- `evaluate_metrics_serverapp`: Server-side evaluation metrics (used for analysis)

## Key Findings to Analyze

When analyzing results, focus on:

1. **Attack Impact**: How do 1 and 2 attackers degrade performance vs baseline?
2. **Data Distribution**: Are attacks more effective on IID or non-IID data?
3. **Strategy Robustness**: Is FedProx more resilient to attacks than FedAvg?
4. **Metric Sensitivity**: Which metrics (Accuracy, F1, Kappa, ROC) show the strongest degradation?

## Customization

### Change Attack Type
Edit `flower2025/task.py`:
```python
def train(net, trainloader, epochs, lr, device, is_attacker=False, attack_type="label_flipping"):
    if is_attacker:
        if attack_type == "label_flipping":
            # Your custom attack here
            labels = torch.randint(0, 10, labels.shape, device=device)
```

### Add More Attackers
Edit `run_experiments.py`:
```python
# 3 attackers (clients 0, 1, and 2)
experiments.append({
    "strategy": "FedAvg",
    "use_iid": True,
    "num_attackers": 3,
    "attacker_ids": [0, 1, 2],
    "attack_type": attack_type,
})
```

### Change Model Architecture
Edit `flower2025/task.py` in the `Net` class.

## Troubleshooting

### Issue: "Dataset download failed"
**Solution**: Ensure internet connection. Dataset auto-downloads on first run.

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `load_data()` function or use CPU only.

### Issue: "Import errors"
**Solution**: Ensure virtual environment is activated and all packages installed.

### Issue: "Experiments take too long"
**Solution**: Reduce `num-server-rounds` in `pyproject.toml` or run fewer experiments.

## Expected Runtime

On typical hardware (CPU: i7, RAM: 16GB):
- Single experiment: ~10-15 minutes
- All 12 experiments: ~2-4 hours

With GPU acceleration:
- Single experiment: ~5-8 minutes
- All 12 experiments: ~1-2 hours

## Citation

This project uses:
- **Flower Framework**: https://flower.ai/
- **Flower Datasets**: https://flower.ai/docs/datasets/
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html

## License

Apache 2.0 (as per Flower framework)

## Contact

For questions or issues, refer to the Flower documentation: https://flower.ai/docs/

---

**Note**: This is an educational project demonstrating federated learning vulnerabilities. Results should be analyzed critically in the context of the specific attack mechanism and experimental setup.
