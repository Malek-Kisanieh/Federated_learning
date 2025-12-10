# Mhd Malek Kisanieh
# Samir Akhalil
"""
Comprehensive analysis and visualization script for FL attack experiments.
Generates comparison plots for all metrics across different scenarios.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


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


def plot_comparison(data_dict, title, ylabel, filename, ylim=None):
    """Create a comparison plot for a specific metric."""
    plt.figure(figsize=(12, 7))
    
    colors = {
        "baseline": "#2ecc71",
        "1attacker": "#e74c3c",
        "2attackers": "#e67e22",
    }
    
    linestyles = {
        "baseline": "-",
        "1attacker": "--",
        "2attackers": "-.",
    }
    
    markers = {
        "baseline": "o",
        "1attacker": "s",
        "2attackers": "^",
    }
    
    for label, data in data_dict.items():
        if data is None:
            continue
        
        # Determine style based on number of attackers
        if "0attackers" in label or "baseline" in label.lower():
            style_key = "baseline"
            display_label = label.replace("0attackers", "Baseline")
        elif "1attacker" in label:
            style_key = "1attacker"
            display_label = label.replace("1attackers", "1 Attacker")
        elif "2attacker" in label:
            style_key = "2attackers"
            display_label = label.replace("2attackers", "2 Attackers")
        else:
            style_key = "baseline"
            display_label = label
        
        plt.plot(
            data["rounds"],
            data[ylabel.lower()],
            label=display_label,
            color=colors.get(style_key, "#3498db"),
            linestyle=linestyles.get(style_key, "-"),
            marker=markers.get(style_key, "o"),
            markevery=5,
            linewidth=2,
            markersize=6,
            alpha=0.8
        )
    
    plt.xlabel("Communication Round", fontsize=12, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.legend(loc="best", fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle="--")
    
    if ylim:
        plt.ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def create_comprehensive_plots():
    """Create comprehensive comparison plots for all scenarios."""
    
    results_dir = Path("results")
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80 + "\n")
    
    # Define all experiment scenarios
    strategies = ["FedAvg", "FedProx"]
    data_dists = ["iid", "non_iid"]
    attacker_counts = [0, 1, 2]
    
    metrics_to_plot = ["Accuracy", "Loss", "Kappa", "F1", "ROC"]
    
    # Part 2.1 and 2.2: Compare baseline vs attackers for each strategy and data distribution
    for strategy in strategies:
        for data_dist in data_dists:
            print(f"\n--- {strategy} with {data_dist.upper()} data ---")
            
            data_dict = {}
            for num_attackers in attacker_counts:
                exp_name = f"{strategy}_{data_dist}_{num_attackers}attackers"
                metrics_file = results_dir / f"metrics_{exp_name}.json"
                
                metrics_data = load_metrics(metrics_file)
                if metrics_data:
                    server_metrics = extract_server_metrics(metrics_data)
                    data_dict[exp_name] = server_metrics
            
            # Create plots for each metric
            for metric in metrics_to_plot:
                plot_title = f"{strategy} - {data_dist.upper()} Data: {metric} Comparison"
                plot_filename = plots_dir / f"{strategy}_{data_dist}_{metric.lower()}_comparison.png"
                
                plot_comparison(
                    data_dict,
                    plot_title,
                    metric,
                    str(plot_filename),
                    ylim=(0, 1) if metric != "Loss" else None
                )
    
    # Cross-strategy comparison: FedAvg vs FedProx
    print(f"\n--- Cross-Strategy Comparisons ---")
    
    for data_dist in data_dists:
        for num_attackers in attacker_counts:
            print(f"\n  Comparing strategies: {data_dist.upper()} with {num_attackers} attacker(s)")
            
            data_dict = {}
            for strategy in strategies:
                exp_name = f"{strategy}_{data_dist}_{num_attackers}attackers"
                metrics_file = results_dir / f"metrics_{exp_name}.json"
                
                metrics_data = load_metrics(metrics_file)
                if metrics_data:
                    server_metrics = extract_server_metrics(metrics_data)
                    data_dict[exp_name] = server_metrics
            
            # Create plots for each metric
            for metric in metrics_to_plot:
                attack_label = "Baseline" if num_attackers == 0 else f"{num_attackers}_Attackers"
                plot_title = f"{data_dist.upper()} Data - {attack_label}: Strategy Comparison ({metric})"
                plot_filename = plots_dir / f"strategy_comparison_{data_dist}_{attack_label}_{metric.lower()}.png"
                
                plot_comparison(
                    data_dict,
                    plot_title,
                    metric,
                    str(plot_filename),
                    ylim=(0, 1) if metric != "Loss" else None
                )
    
    # Create summary dashboard
    print(f"\n--- Creating Summary Dashboard ---")
    create_summary_dashboard()
    
    print("\n" + "="*80)
    print(f"All plots saved in '{plots_dir}/' directory")
    print("="*80 + "\n")


def create_summary_dashboard():
    """Create a comprehensive dashboard showing final metrics for all experiments."""
    
    results_dir = Path("results")
    plots_dir = Path("plots")
    
    # Collect final round metrics for all experiments
    strategies = ["FedAvg", "FedProx"]
    data_dists = ["iid", "non_iid"]
    attacker_counts = [0, 1, 2]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Final Metrics Comparison Across All Experiments", 
                 fontsize=16, fontweight="bold", y=0.995)
    
    metrics_to_plot = ["Accuracy", "F1", "Kappa", "ROC"]
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        x_pos = 0
        x_labels = []
        x_ticks = []
        
        for strategy in strategies:
            for data_dist in data_dists:
                values = []
                labels = []
                
                for num_attackers in attacker_counts:
                    exp_name = f"{strategy}_{data_dist}_{num_attackers}attackers"
                    metrics_file = results_dir / f"metrics_{exp_name}.json"
                    
                    metrics_data = load_metrics(metrics_file)
                    if metrics_data:
                        server_metrics = extract_server_metrics(metrics_data)
                        if server_metrics and len(server_metrics[metric.lower()]) > 0:
                            final_value = server_metrics[metric.lower()][-1]
                            values.append(final_value)
                            
                            if num_attackers == 0:
                                labels.append("Base")
                            else:
                                labels.append(f"{num_attackers}A")
                
                if values:
                    colors = ["#2ecc71", "#e74c3c", "#e67e22"][:len(values)]
                    positions = [x_pos + i * 0.25 for i in range(len(values))]
                    
                    bars = ax.bar(positions, values, width=0.2, label=labels, color=colors, alpha=0.8)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom', fontsize=8)
                    
                    x_ticks.append(x_pos + 0.25)
                    x_labels.append(f"{strategy}\n{data_dist.upper()}")
                    
                    x_pos += 1
        
        ax.set_ylabel(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_title(f"Final {metric} Values", fontsize=12, fontweight="bold")
    
    # Add legend
    handles = [
        plt.Rectangle((0,0),1,1, color="#2ecc71", alpha=0.8, label="Baseline"),
        plt.Rectangle((0,0),1,1, color="#e74c3c", alpha=0.8, label="1 Attacker"),
        plt.Rectangle((0,0),1,1, color="#e67e22", alpha=0.8, label="2 Attackers"),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              ncol=3, fontsize=11, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    dashboard_file = plots_dir / "summary_dashboard.png"
    plt.savefig(dashboard_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {dashboard_file}")
    plt.close()


def generate_analysis_report():
    """Generate a text-based analysis report."""
    
    results_dir = Path("results")
    report_file = Path("analysis_report.txt")
    
    print("\n--- Generating Analysis Report ---")
    
    with open(report_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("FEDERATED LEARNING ATTACK ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        strategies = ["FedAvg", "FedProx"]
        data_dists = ["iid", "non_iid"]
        attacker_counts = [0, 1, 2]
        
        for strategy in strategies:
            f.write(f"\n{'='*80}\n")
            f.write(f"STRATEGY: {strategy}\n")
            f.write(f"{'='*80}\n")
            
            for data_dist in data_dists:
                f.write(f"\n--- {data_dist.upper()} Data Distribution ---\n\n")
                
                for num_attackers in attacker_counts:
                    exp_name = f"{strategy}_{data_dist}_{num_attackers}attackers"
                    metrics_file = results_dir / f"metrics_{exp_name}.json"
                    
                    metrics_data = load_metrics(metrics_file)
                    if metrics_data:
                        server_metrics = extract_server_metrics(metrics_data)
                        if server_metrics:
                            attack_label = "Baseline" if num_attackers == 0 else f"{num_attackers} Attacker(s)"
                            
                            f.write(f"  {attack_label}:\n")
                            f.write(f"    Final Accuracy: {server_metrics['accuracy'][-1]:.4f}\n")
                            f.write(f"    Final F1 Score:  {server_metrics['f1'][-1]:.4f}\n")
                            f.write(f"    Final Kappa:     {server_metrics['kappa'][-1]:.4f}\n")
                            f.write(f"    Final ROC-AUC:   {server_metrics['roc'][-1]:.4f}\n")
                            f.write(f"    Final Loss:      {server_metrics['loss'][-1]:.4f}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS COMPLETE\n")
        f.write("="*80 + "\n")
    
    print(f"Saved: {report_file}")
    
    # Print report to console
    with open(report_file, "r") as f:
        print("\n" + f.read())


def main():
    """Main entry point for analysis."""
    
    # Check if results directory exists
    if not Path("results").exists():
        print("Error: 'results' directory not found.")
        print("Please run 'python run_experiments.py' first to generate results.")
        return
    
    # Check if any results files exist
    results_files = list(Path("results").glob("metrics_*.json"))
    if not results_files:
        print("Error: No results files found in 'results' directory.")
        print("Please run 'python run_experiments.py' first to generate results.")
        return
    
    print(f"\nFound {len(results_files)} experiment results")
    
    # Generate plots
    create_comprehensive_plots()
    
    # Generate analysis report
    generate_analysis_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - plots/: Contains all comparison plots")
    print("  - analysis_report.txt: Text-based analysis report")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
