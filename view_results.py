"""
Quick results viewer - Display summary of completed experiments
"""

import json
from pathlib import Path
from tabulate import tabulate


def load_and_summarize():
    """Load all results and display summary table."""
    
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("\n❌ No results directory found.")
        print("Run 'python test_setup.py' or 'python run_experiments.py' first.\n")
        return
    
    results_files = list(results_dir.glob("metrics_*.json"))
    
    if not results_files:
        print("\n❌ No results files found in results/ directory.")
        print("Run experiments first.\n")
        return
    
    print(f"\n{'-'*60}")
    print(f"EXPERIMENT RESULTS SUMMARY")
    print(f"{'-'*60}\n")
    print(f"Found {len(results_files)} completed experiments\n")
    
    # Collect all results
    data = []
    
    for results_file in sorted(results_files):
        try:
            with open(results_file, "r") as f:
                metrics = json.load(f)
            
            config = metrics.get("experiment_config", {})
            eval_server = metrics.get("evaluate_metrics_serverapp", {})
            
            if not eval_server:
                continue
            
            # Get final round metrics
            last_round = max([int(k) for k in eval_server.keys()])
            final_metrics = eval_server[str(last_round)]
            
            data.append([
                config.get("strategy", "N/A"),
                config.get("data_distribution", "N/A"),
                config.get("num_attackers", "N/A"),
                f"{final_metrics['accuracy']:.4f}",
                f"{final_metrics['f1']:.4f}",
                f"{final_metrics['kappa']:.4f}",
                f"{final_metrics['roc']:.4f}",
                f"{final_metrics['loss']:.4f}",
            ])
        
        except Exception as e:
            print(f"Warning: Could not parse {results_file.name}: {e}")
    
    if not data:
        print("❌ No valid results found.\n")
        return
    
    # Create table
    headers = ["Strategy", "Data", "Attackers", "Accuracy", "F1", "Kappa", "ROC", "Loss"]
    
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    # Calculate some statistics
    print(f"\n{'-'*60}")
    print("KEY OBSERVATIONS")
    print(f"{'-'*60}\n")
    
    # Group by strategy and data distribution
    for strategy in ["FedAvg", "FedProx"]:
        for data_dist in ["iid", "non_iid"]:
            matching = [row for row in data if row[0] == strategy and row[1] == data_dist]
            
            if len(matching) >= 3:
                baseline = next((row for row in matching if row[2] == 0), None)
                one_attacker = next((row for row in matching if row[2] == 1), None)
                two_attackers = next((row for row in matching if row[2] == 2), None)
                
                if baseline and one_attacker and two_attackers:
                    print(f"{strategy} - {data_dist.upper()}:")
                    baseline_acc = float(baseline[3])
                    one_att_acc = float(one_attacker[3])
                    two_att_acc = float(two_attackers[3])
                    
                    print(f"  Baseline Accuracy:     {baseline_acc:.4f}")
                    print(f"  1 Attacker Impact:     {baseline_acc - one_att_acc:+.4f} ({((baseline_acc - one_att_acc) / baseline_acc * 100):.1f}% degradation)")
                    print(f"  2 Attackers Impact:    {baseline_acc - two_att_acc:+.4f} ({((baseline_acc - two_att_acc) / baseline_acc * 100):.1f}% degradation)")
                    print()


if __name__ == "__main__":
    try:
        load_and_summarize()
    except ImportError:
        print("\n⚠️  'tabulate' package not found. Installing...")
        print("Run: pip install tabulate")
        print("\nOr use the basic version:\n")
        
        # Fallback without tabulate
        results_dir = Path("results")
        if results_dir.exists():
            results_files = list(results_dir.glob("metrics_*.json"))
            print(f"Found {len(results_files)} results files:")
            for f in sorted(results_files):
                print(f"  - {f.name}")
        else:
            print("No results directory found.")
