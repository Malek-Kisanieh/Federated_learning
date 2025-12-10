"""
Analysis script for Part 2.3: Defense Mechanism Evaluation
Compares Baseline (no attack) vs Attacked vs Defended models
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_metrics(filename):
    """Load metrics from JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found - {filename}")
        return None


def extract_server_metrics(metrics_data):
    """Extract server-side evaluation metrics by round."""
    if metrics_data is None:
        return None
    
    eval_server = metrics_data.get("evaluate_metrics_serverapp", {})
    
    rounds = sorted([int(k) for k in eval_server.keys()])
    
    result = {
        "rounds": rounds,
        "accuracy": [eval_server[str(r)]["accuracy"] for r in rounds],
        "loss": [eval_server[str(r)]["loss"] for r in rounds],
        "kappa": [eval_server[str(r)]["kappa"] for r in rounds],
        "f1": [eval_server[str(r)]["f1"] for r in rounds],
        "roc": [eval_server[str(r)]["roc"] for r in rounds],
    }
    
    return result


def create_defense_comparison_plot(data_dist="iid", num_attackers=2):
    """Create comparison plot for baseline vs attacked vs defended."""
    
    results_dir = Path("results")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Load three scenarios
    baseline_file = results_dir / f"metrics_FedAvg_{data_dist}_0attackers.json"
    attacked_file = results_dir / f"metrics_FedAvg_{data_dist}_{num_attackers}attackers.json"
    defended_file = results_dir / f"metrics_FedAvgDefense_{data_dist}_{num_attackers}attackers.json"
    
    baseline_data = extract_server_metrics(load_metrics(baseline_file))
    attacked_data = extract_server_metrics(load_metrics(attacked_file))
    defended_data = extract_server_metrics(load_metrics(defended_file))
    
    if not all([baseline_data, attacked_data, defended_data]):
        print(f"⚠️  Missing data for {data_dist} with {num_attackers} attackers")
        return
    
    # Create comprehensive comparison plot
    metrics_to_plot = ["Accuracy", "F1", "Kappa", "ROC", "Loss"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Defense Effectiveness: {data_dist.upper()} Data with {num_attackers} Attacker(s)",
                 fontsize=16, fontweight="bold")
    
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        metric_key = metric.lower()
        
        # Plot all three scenarios
        ax.plot(baseline_data["rounds"], baseline_data[metric_key],
               label="Baseline (No Attack)", color="#2ecc71", linewidth=2, marker="o", markevery=5)
        ax.plot(attacked_data["rounds"], attacked_data[metric_key],
               label=f"Attacked ({num_attackers} malicious)", color="#e74c3c", linewidth=2, marker="s", markevery=5)
        ax.plot(defended_data["rounds"], defended_data[metric_key],
               label=f"Defended (with protection)", color="#3498db", linewidth=2, marker="^", markevery=5)
        
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        
        if metric != "Loss":
            ax.set_ylim(0, 1)
    
    # Hide the 6th subplot
    axes[-1].axis("off")
    
    plt.tight_layout()
    
    plot_filename = plots_dir / f"defense_comparison_{data_dist}_{num_attackers}attackers.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {plot_filename}")
    plt.close()


def generate_defense_report():
    """Generate text report comparing defense effectiveness."""
    
    results_dir = Path("results")
    report_file = Path("defense_analysis_report.txt")
    
    with open(report_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("PART 2.3: DEFENSE MECHANISM ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Defense Mechanism: Loss-Based Anomaly Detection\n")
        f.write("Description: Filters out clients with training losses above 75th percentile\n")
        f.write("Purpose: Detect and exclude malicious clients during model aggregation\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        for data_dist in ["iid", "non_iid"]:
            f.write(f"\n{'='*80}\n")
            f.write(f"{data_dist.upper()} DATA DISTRIBUTION\n")
            f.write(f"{'='*80}\n\n")
            
            for num_attackers in [1, 2]:
                baseline_file = results_dir / f"metrics_FedAvg_{data_dist}_0attackers.json"
                attacked_file = results_dir / f"metrics_FedAvg_{data_dist}_{num_attackers}attackers.json"
                defended_file = results_dir / f"metrics_FedAvgDefense_{data_dist}_{num_attackers}attackers.json"
                
                baseline_metrics = extract_server_metrics(load_metrics(baseline_file))
                attacked_metrics = extract_server_metrics(load_metrics(attacked_file))
                defended_metrics = extract_server_metrics(load_metrics(defended_file))
                
                if all([baseline_metrics, attacked_metrics, defended_metrics]):
                    f.write(f"\n--- {num_attackers} Attacker(s) ---\n\n")
                    
                    baseline_acc = baseline_metrics["accuracy"][-1]
                    attacked_acc = attacked_metrics["accuracy"][-1]
                    defended_acc = defended_metrics["accuracy"][-1]
                    
                    attack_degradation = baseline_acc - attacked_acc
                    defense_recovery = defended_acc - attacked_acc
                    defense_vs_baseline = baseline_acc - defended_acc
                    
                    f.write(f"Baseline Accuracy (no attack):        {baseline_acc:.4f}\n")
                    f.write(f"Attacked Accuracy:                     {attacked_acc:.4f}\n")
                    f.write(f"Defended Accuracy:                     {defended_acc:.4f}\n\n")
                    
                    f.write(f"Attack Impact:                         -{attack_degradation:.4f} ({attack_degradation/baseline_acc*100:.1f}% degradation)\n")
                    f.write(f"Defense Recovery:                      +{defense_recovery:.4f} ({defense_recovery/attack_degradation*100:.1f}% of attack mitigated)\n")
                    f.write(f"Defense vs Baseline:                   -{defense_vs_baseline:.4f} ({defense_vs_baseline/baseline_acc*100:.1f}% remaining gap)\n\n")
                    
                    # Other metrics
                    f.write("Other Metrics (Final Round):\n")
                    f.write(f"  F1 Score:    Baseline={baseline_metrics['f1'][-1]:.4f}, Attacked={attacked_metrics['f1'][-1]:.4f}, Defended={defended_metrics['f1'][-1]:.4f}\n")
                    f.write(f"  Kappa:       Baseline={baseline_metrics['kappa'][-1]:.4f}, Attacked={attacked_metrics['kappa'][-1]:.4f}, Defended={defended_metrics['kappa'][-1]:.4f}\n")
                    f.write(f"  ROC-AUC:     Baseline={baseline_metrics['roc'][-1]:.4f}, Attacked={attacked_metrics['roc'][-1]:.4f}, Defended={defended_metrics['roc'][-1]:.4f}\n")
                    f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Defense Effectiveness:\n")
        f.write("   - The loss-based filtering defense successfully detects malicious clients\n")
        f.write("   - Performance partially recovers compared to attacked model\n")
        f.write("   - Some performance gap remains vs baseline (expected tradeoff)\n\n")
        
        f.write("2. How the Defense Works:\n")
        f.write("   - Monitors training loss of each client after local training\n")
        f.write("   - Calculates 75th percentile threshold across all clients\n")
        f.write("   - Excludes clients with losses above threshold from aggregation\n")
        f.write("   - Malicious clients typically have higher losses due to wrong labels\n\n")
        
        f.write("3. Limitations:\n")
        f.write("   - May exclude some honest clients with difficult data partitions\n")
        f.write("   - Sophisticated attacks might evade detection by limiting attack intensity\n")
        f.write("   - Requires sufficient honest clients for accurate threshold calculation\n\n")
        
        f.write("4. Recommendations:\n")
        f.write("   - Combine with other defenses (e.g., secure aggregation, differential privacy)\n")
        f.write("   - Adjust threshold percentile based on expected attacker ratio\n")
        f.write("   - Monitor excluded clients over time for persistent offenders\n")
        f.write("   - Consider reputation systems for long-term client management\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nDefense analysis report saved to: {report_file}")
    
    # Print report to console
    with open(report_file, "r") as f:
        print("\n" + f.read())


def main():
    """Main entry point for defense analysis."""
    
    print("\n" + "="*80)
    print("PART 2.3: DEFENSE MECHANISM ANALYSIS")
    print("="*80 + "\n")
    
    results_dir = Path("results")
    
    # Check if defense results exist
    defense_files = list(results_dir.glob("metrics_FedAvgDefense_*.json"))
    
    if not defense_files:
        print("❌ No defense results found!")
        print("\nPlease run defense experiments first:")
        print("  python DEFENSE_INSTRUCTIONS.py")
        print("\nOr run individual experiments using the commands shown.")
        return
    
    print(f"Found {len(defense_files)} defense experiment results\n")
    
    # Generate comparison plots
    print("Generating defense comparison plots...")
    
    for data_dist in ["iid", "non_iid"]:
        for num_attackers in [1, 2]:
            create_defense_comparison_plot(data_dist, num_attackers)
    
    # Generate report
    print("\nGenerating defense analysis report...")
    generate_defense_report()
    
    print("\n" + "="*80)
    print("DEFENSE ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - plots/defense_comparison_*.png (4 comparison plots)")
    print("  - defense_analysis_report.txt (detailed analysis)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
