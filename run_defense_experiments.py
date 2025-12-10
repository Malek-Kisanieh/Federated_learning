"""
Run defense experiments (Part 2.3)
Compare baseline, attacked, and defended models
"""

import subprocess
from pathlib import Path
import json


def run_defense_experiment(use_iid=True, num_attackers=0, attacker_ids=None):
    """Run a defense experiment."""
    
    if attacker_ids is None:
        attacker_ids = []
    
    attacker_ids_str = ",".join(map(str, attacker_ids)) if attacker_ids else "none"
    data_dist = "iid" if use_iid else "non_iid"
    attack_type = "label_flipping" if num_attackers > 0 else "none"
    
    experiment_name = f"FedAvgDefense_{data_dist}_{num_attackers}attackers"
    
    print(f"\n{'='*80}")
    print(f"Running DEFENSE experiment: {experiment_name}")
    print(f"Data: {data_dist}, Attackers: {num_attackers}")
    print(f"{'='*80}\n")
    
    # Build config string with defense parameters
    config_parts = [
        "num-server-rounds=50",
        f"use-iid={str(use_iid).lower()}",
        f'attacker-ids="{attacker_ids_str}"',
        f'attack-type="{attack_type}"',
        "loss-threshold-percentile=75",  # Defense parameter
    ]
    
    config_string = " ".join(config_parts)
    
    # Use defense server app
    cmd = [
        "flwr", "run", ".",
        "--run-config",
        config_string,
    ]
    
    # Override to use defense server app
    # We need to temporarily update pyproject.toml or use environment variable
    # For now, we'll create a custom command
    
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent,
            env={**subprocess.os.environ, "FLWR_SERVER_APP": "flower2025.server_app_defense:app"}
        )
        print(f"✓ Defense experiment {experiment_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Defense experiment {experiment_name} failed!")
        return False


def main():
    """Run defense experiments for Part 2.3"""
    
    print("\n" + "="*80)
    print("PART 2.3: DEFENSE MECHANISM EXPERIMENTS")
    print("Defense: Loss-Based Anomaly Detection and Client Filtering")
    print("="*80)
    
    # Run defense experiments for both IID and non-IID
    # Compare: Baseline (0 attackers), 1 attacker, 2 attackers - all WITH defense
    
    experiments = [
        # IID experiments with defense
        {"use_iid": True, "num_attackers": 0, "attacker_ids": []},
        {"use_iid": True, "num_attackers": 1, "attacker_ids": [0]},
        {"use_iid": True, "num_attackers": 2, "attacker_ids": [0, 1]},
        
        # Non-IID experiments with defense
        {"use_iid": False, "num_attackers": 0, "attacker_ids": []},
        {"use_iid": False, "num_attackers": 1, "attacker_ids": [0]},
        {"use_iid": False, "num_attackers": 2, "attacker_ids": [0, 1]},
    ]
    
    print(f"\nTotal defense experiments to run: {len(experiments)}")
    print("These will be compared against the baseline FedAvg experiments\n")
    
    success_count = 0
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n--- Defense Experiment {i}/{len(experiments)} ---")
        
        success = run_defense_experiment(
            use_iid=exp["use_iid"],
            num_attackers=exp["num_attackers"],
            attacker_ids=exp["attacker_ids"]
        )
        
        if success:
            success_count += 1
    
    print("\n" + "="*80)
    print("DEFENSE EXPERIMENTS SUMMARY")
    print("="*80)
    print(f"\nCompleted: {success_count}/{len(experiments)}")
    print("\nNext step: Run 'python analyze_defense.py' to compare results")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
