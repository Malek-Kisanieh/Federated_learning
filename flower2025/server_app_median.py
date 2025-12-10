"""flower2025: A Flower / PyTorch app with Median Aggregation defense."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, Context
from flower2025.task import Net, load_data, test as test_fn
import json
import os
import numpy as np
from collections import OrderedDict

# Create ServerApp with median defense
app = ServerApp()


class FedMedian(FedAvg):
    """FedMedian: Uses coordinate-wise median instead of mean for aggregation.
    
    This is a Byzantine-robust aggregation strategy that is more resilient
    to outliers and malicious updates than simple averaging.
    """
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate using coordinate-wise median instead of weighted average."""
        
        if not results:
            return None, {}
        
        # Extract weights from all clients
        weights_list = []
        for client_proxy, fit_res in results:
            weights = fit_res.arrays.to_torch_state_dict()
            weights_list.append(weights)
        
        if not weights_list:
            return None, {}
        
        print(f"  🛡️  Defense: Using median aggregation with {len(weights_list)} clients")
        
        # Perform coordinate-wise median aggregation
        aggregated_weights = OrderedDict()
        
        for key in weights_list[0].keys():
            # Stack all client weights for this layer
            layer_weights = torch.stack([w[key] for w in weights_list])
            
            # Compute median across clients (dimension 0)
            median_weights = torch.median(layer_weights, dim=0)[0]
            
            aggregated_weights[key] = median_weights
        
        # Convert back to ArrayRecord
        aggregated_arrays = ArrayRecord(aggregated_weights)
        
        # Collect metrics
        metrics = {}
        for _, fit_res in results:
            for key, val in fit_res.metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(val)
        
        # Average the metrics
        aggregated_metrics = {
            key: sum(values) / len(values) 
            for key, values in metrics.items()
        }
        
        return aggregated_arrays, aggregated_metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp with median defense."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    use_iid: bool = context.run_config.get("use-iid", True)
    
    # For saving results with unique filename
    attacker_ids_str = context.run_config.get("attacker-ids", "none")
    if attacker_ids_str and attacker_ids_str.strip() and attacker_ids_str.lower() != "none":
        attacker_ids = [int(x.strip()) for x in attacker_ids_str.split(",")]
    else:
        attacker_ids = []
    attack_type = context.run_config.get("attack-type", "none")
    num_attackers = len(attacker_ids)
    data_dist = "iid" if use_iid else "non_iid"
    
    # Create unique experiment name with defense marker
    experiment_name = f"FedMedian_{data_dist}_{num_attackers}attackers"
    
    print(f"\n{'='*80}")
    print(f"Starting experiment with MEDIAN DEFENSE: {experiment_name}")
    print(f"Strategy: Coordinate-Wise Median Aggregation")
    print(f"Data: {data_dist}, Attackers: {num_attackers}")
    print(f"Attacker IDs: {attacker_ids}, Attack type: {attack_type}")
    print(f"Defense: Byzantine-robust median aggregation")
    print(f"{'='*80}\n")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Load centralized test data for server-side evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, testloader = load_data(partition_id=0, num_partitions=10, use_iid=use_iid)

    # Define server-side evaluation function
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> dict:
        """Evaluate global model on centralized test set."""
        net = Net()
        net.load_state_dict(arrays.to_torch_state_dict())
        loss, accuracy, kappa, f1, roc = test_fn(net, testloader, device)
        
        roc_value = roc if roc is not None else 0.0
        print(f"Round {server_round:2d} | Loss: {loss:.4f} | Acc: {accuracy:.4f} | Kappa: {kappa:.4f} | F1: {f1:.4f} | ROC: {roc_value:.4f}")
        
        return {
            "loss": loss,
            "accuracy": accuracy,
            "kappa": kappa,
            "f1": f1,
            "roc": roc_value
        }

    # Initialize FedMedian strategy
    strategy = FedMedian(fraction_train=fraction_train)
    print(f"Using FedMedian (Coordinate-Wise Median Aggregation)")

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    # Save metrics with experiment-specific filename
    metrics_to_save = {
        "experiment_config": {
            "strategy": "FedMedian",
            "data_distribution": data_dist,
            "num_attackers": num_attackers,
            "attacker_ids": attacker_ids,
            "attack_type": attack_type,
            "num_rounds": num_rounds,
            "num_clients": 5,
            "defense_mechanism": "coordinate_wise_median",
        },
        "train_metrics_clientapp": {k: dict(v) for k, v in result.train_metrics_clientapp.items()},
        "evaluate_metrics_clientapp": {k: dict(v) for k, v in result.evaluate_metrics_clientapp.items()},
        "evaluate_metrics_serverapp": {k: dict(v) for k, v in result.evaluate_metrics_serverapp.items()},
    }

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    metrics_filename = f"results/metrics_{experiment_name}.json"
    with open(metrics_filename, "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print(f"\nMetrics saved to: {metrics_filename}")

    # Save final model to disk
    model_filename = f"results/final_model_{experiment_name}.pt"
    print(f"Saving final model to: {model_filename}")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, model_filename)
    
    print(f"\n{'='*80}")
    print(f"Experiment {experiment_name} completed!")
    print(f"{'='*80}\n")
