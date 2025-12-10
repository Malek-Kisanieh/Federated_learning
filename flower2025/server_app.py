# Mhd Malek Kisanieh
# Samir Akhalil
"""flower2025: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx
from flwr.app import ArrayRecord, Context
from flower2025.task import Net, load_data, test as test_fn
import json
import os

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    use_iid: bool = context.run_config.get("use-iid", True)
    strategy_name: str = context.run_config.get("strategy", "FedAvg")
    proximal_mu: float = context.run_config.get("proximal-mu", 0.1)
    
    # For saving results with unique filename
    attacker_ids_str = context.run_config.get("attacker-ids", "none")
    if attacker_ids_str and attacker_ids_str.strip() and attacker_ids_str.lower() != "none":
        attacker_ids = [int(x.strip()) for x in attacker_ids_str.split(",")]
    else:
        attacker_ids = []
    attack_type = context.run_config.get("attack-type", "none")
    num_attackers = len(attacker_ids)
    data_dist = "iid" if use_iid else "non_iid"
    
    # Create unique experiment name
    experiment_name = f"{strategy_name}_{data_dist}_{num_attackers}attackers"
    
    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Strategy: {strategy_name}, Data: {data_dist}, Attackers: {num_attackers}")
    print(f"Attacker IDs: {attacker_ids}, Attack type: {attack_type}")
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

    # Initialize strategy based on config
    if strategy_name.lower() == "fedprox":
        strategy = FedProx(
            fraction_train=fraction_train,
            proximal_mu=proximal_mu,
        )
        print(f"Using FedProx with proximal_mu={proximal_mu}")
    else:
        strategy = FedAvg(fraction_train=fraction_train)
        print(f"Using FedAvg")

    # Start strategy, run for `num_rounds` with server-side evaluation
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
            "strategy": strategy_name,
            "data_distribution": data_dist,
            "num_attackers": num_attackers,
            "attacker_ids": attacker_ids,
            "attack_type": attack_type,
            "num_rounds": num_rounds,
            "num_clients": 5,
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
