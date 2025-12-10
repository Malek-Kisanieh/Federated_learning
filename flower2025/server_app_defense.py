"""flower2025: A Flower / PyTorch app with defense mechanism."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, Context
from flower2025.task import Net, load_data, test as test_fn
import json
import os
import numpy as np

# Create ServerApp with defense
app = ServerApp()


class FedAvgDefense(FedAvg):
    """FedAvg with defense mechanism: filters clients with anomalous losses."""
    
    def __init__(self, *args, loss_threshold_percentile=75, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_threshold_percentile = loss_threshold_percentile
        self.excluded_clients = []
        
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate with anomaly detection on training losses."""
        
        if not results:
            return None, {}
        
        # Extract losses from results
        losses = []
        valid_results = []
        
        for client_proxy, fit_res in results:
            train_loss = fit_res.metrics.get("train_loss", 0)
            losses.append(train_loss)
            valid_results.append((client_proxy, fit_res, train_loss))
        
        # Calculate threshold (e.g., 75th percentile)
        if len(losses) > 2:
            threshold = np.percentile(losses, self.loss_threshold_percentile)
            
            # Filter out clients with suspiciously high losses
            filtered_results = []
            excluded_count = 0
            
            for client_proxy, fit_res, loss in valid_results:
                if loss <= threshold:
                    filtered_results.append((client_proxy, fit_res))
                else:
                    excluded_count += 1
                    print(f"  ⚠️  Excluding client with high loss: {loss:.4f} (threshold: {threshold:.4f})")
            
            if excluded_count > 0:
                print(f"  🛡️  Defense: Excluded {excluded_count}/{len(results)} clients in round {server_round}")
            
            # Use filtered results for aggregation
            if filtered_results:
                results = filtered_results
            else:
                print(f"  ⚠️  All clients excluded - using all results as fallback")
        
        # Call parent aggregation with filtered results
        return super().aggregate_fit(server_round, results, failures)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp with defense."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    use_iid: bool = context.run_config.get("use-iid", True)
    
    # Defense parameters
    loss_threshold_percentile: int = context.run_config.get("loss-threshold-percentile", 75)
    
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
    experiment_name = f"FedAvgDefense_{data_dist}_{num_attackers}attackers"
    
    print(f"\n{'='*80}")
    print(f"Starting experiment with DEFENSE: {experiment_name}")
    print(f"Strategy: FedAvg + Loss-Based Defense")
    print(f"Data: {data_dist}, Attackers: {num_attackers}")
    print(f"Attacker IDs: {attacker_ids}, Attack type: {attack_type}")
    print(f"Defense: Loss threshold at {loss_threshold_percentile}th percentile")
    print(f"{'='*80}\n")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Load centralized test data for server-side evaluation
    # Force CPU due to GPU compatibility issues in WSL
    device = torch.device("cpu")
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

    # Initialize FedAvg with defense
    strategy = FedAvgDefense(
        fraction_train=fraction_train,
        loss_threshold_percentile=loss_threshold_percentile,
    )
    print(f"Using FedAvg with Loss-Based Defense (threshold: {loss_threshold_percentile}th percentile)")

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
            "strategy": "FedAvgDefense",
            "data_distribution": data_dist,
            "num_attackers": num_attackers,
            "attacker_ids": attacker_ids,
            "attack_type": attack_type,
            "num_rounds": num_rounds,
            "num_clients": 5,
            "defense_mechanism": "loss_based_filtering",
            "loss_threshold_percentile": loss_threshold_percentile,
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
