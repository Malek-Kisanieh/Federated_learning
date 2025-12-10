"""
Comprehensive experiment runner for FL attack simulation.
This script runs all required experiments for Part 2.1 and 2.2:
- Baseline (no attackers)
- 1 attacker
- 2 attackers
- IID vs non-IID data
- FedAvg vs FedProx strategies
"""

import subprocess
import json
import os
from pathlib import Path


def run_flower_simulation(
    strategy="FedAvg",
    use_iid=True,
    num_attackers=0,
    attacker_ids=None,
    attack_type="label_flipping",
    proximal_mu=0.1,
    num_rounds=50,
    num_clients=5,
):
    """Run a single Flower simulation experiment."""
    
    if attacker_ids is None:
        attacker_ids = []
    
    data_dist = "iid" if use_iid else "non_iid"
    experiment_name = f"{strategy}_{data_dist}_{num_attackers}attackers"
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"Strategy: {strategy}, Data: {data_dist}, Attackers: {num_attackers}")
    print(f"Attacker IDs: {attacker_ids}, Attack type: {attack_type}")
    print(f"{'='*80}\n")
    
    # Build the flwr run command with all necessary config overrides
    attacker_ids_str = ",".join(map(str, attacker_ids)) if attacker_ids else "none"
    
    config_parts = [
        f"num-server-rounds={num_rounds}",
        f"use-iid={str(use_iid).lower()}",
        f'strategy="{strategy}"',
        f'attacker-ids="{attacker_ids_str}"',
        f'attack-type="{attack_type}"',
    ]
    
    if strategy.lower() == "fedprox":
        config_parts.append(f"proximal-mu={proximal_mu}")
    
    config_string = " ".join(config_parts)
    
    cmd = [
        "flwr", "run", ".",
        "--run-config",
        config_string,
    ]
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        print(f"✓ Experiment {experiment_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment {experiment_name} failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Run all experiments for Part 2.1 and 2.2"""
    
    print("\n" + "="*80)
    print("FEDERATED LEARNING ATTACK SIMULATION - COMPREHENSIVE EXPERIMENT SUITE")
    print("="*80)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    experiments_run = []
    experiments_failed = []
    
    # Attack configuration
    attack_type = "label_flipping"
    
    # Define all experiment configurations
    experiments = []
    
    # Part 2.1: FedAvg experiments
    print("\n" + "="*80)
    print("PART 2.1: FedAvg with IID and Non-IID Data")
    print("="*80)
    
    for use_iid in [True, False]:
        data_type = "IID" if use_iid else "Non-IID"
        
        print(f"\n--- {data_type} Data Distribution ---")
        
        # Baseline (no attackers)
        experiments.append({
            "name": f"FedAvg_{data_type}_Baseline",
            "strategy": "FedAvg",
            "use_iid": use_iid,
            "num_attackers": 0,
            "attacker_ids": [],
            "attack_type": attack_type,
        })
        
        # 1 attacker (client 0)
        experiments.append({
            "name": f"FedAvg_{data_type}_1Attacker",
            "strategy": "FedAvg",
            "use_iid": use_iid,
            "num_attackers": 1,
            "attacker_ids": [0],
            "attack_type": attack_type,
        })
        
        # 2 attackers (clients 0 and 1)
        experiments.append({
            "name": f"FedAvg_{data_type}_2Attackers",
            "strategy": "FedAvg",
            "use_iid": use_iid,
            "num_attackers": 2,
            "attacker_ids": [0, 1],
            "attack_type": attack_type,
        })
    
    # Part 2.2: FedProx experiments
    print("\n" + "="*80)
    print("PART 2.2: FedProx with IID and Non-IID Data")
    print("="*80)
    
    for use_iid in [True, False]:
        data_type = "IID" if use_iid else "Non-IID"
        
        print(f"\n--- {data_type} Data Distribution ---")
        
        # Baseline (no attackers)
        experiments.append({
            "name": f"FedProx_{data_type}_Baseline",
            "strategy": "FedProx",
            "use_iid": use_iid,
            "num_attackers": 0,
            "attacker_ids": [],
            "attack_type": attack_type,
            "proximal_mu": 0.1,
        })
        
        # 1 attacker (client 0)
        experiments.append({
            "name": f"FedProx_{data_type}_1Attacker",
            "strategy": "FedProx",
            "use_iid": use_iid,
            "num_attackers": 1,
            "attacker_ids": [0],
            "attack_type": attack_type,
            "proximal_mu": 0.1,
        })
        
        # 2 attackers (clients 0 and 1)
        experiments.append({
            "name": f"FedProx_{data_type}_2Attackers",
            "strategy": "FedProx",
            "use_iid": use_iid,
            "num_attackers": 2,
            "attacker_ids": [0, 1],
            "attack_type": attack_type,
            "proximal_mu": 0.1,
        })
    
    # Run all experiments
    print(f"\n{'='*80}")
    print(f"Total experiments to run: {len(experiments)}")
    print(f"{'='*80}\n")
    
    for i, exp in enumerate(experiments, 1):
        print(f"\nExperiment {i}/{len(experiments)}: {exp['name']}")
        
        success = run_flower_simulation(
            strategy=exp["strategy"],
            use_iid=exp["use_iid"],
            num_attackers=exp["num_attackers"],
            attacker_ids=exp["attacker_ids"],
            attack_type=exp["attack_type"],
            proximal_mu=exp.get("proximal_mu", 0.1),
        )
        
        if success:
            experiments_run.append(exp["name"])
        else:
            experiments_failed.append(exp["name"])
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Successfully completed: {len(experiments_run)}")
    print(f"Failed: {len(experiments_failed)}")
    
    if experiments_run:
        print("\n✓ Completed experiments:")
        for exp in experiments_run:
            print(f"  - {exp}")
    
    if experiments_failed:
        print("\n✗ Failed experiments:")
        for exp in experiments_failed:
            print(f"  - {exp}")
    
    print("\n" + "="*80)
    print("All results saved in 'results/' directory")
    print("Run 'python analyze_results.py' to generate comparison plots")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
