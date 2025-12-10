#!/usr/bin/env python3
"""
Helper script to run defense experiments (Part 2.3 and 2.4).
Supports two defense mechanisms:
1. Loss-based filtering (FedAvgDefense)
2. Median aggregation (FedMedian)
"""

import subprocess
import shutil
from pathlib import Path
import sys


def run_defense_experiment(use_iid=True, num_attackers=2, attacker_ids=None, defense_type="loss"):
    """Run a single defense experiment.
    
    Args:
        use_iid: Whether to use IID data distribution
        num_attackers: Number of attacking clients
        attacker_ids: List of attacker client IDs
        defense_type: "loss" for loss-based filtering, "median" for median aggregation
    """
    
    if attacker_ids is None:
        attacker_ids = [0, 1] if num_attackers == 2 else [0]
    
    attacker_ids_str = ",".join(map(str, attacker_ids)) if attacker_ids else "none"
    data_dist = "IID" if use_iid else "Non-IID"
    defense_name = "Loss-Based Filtering" if defense_type == "loss" else "Median Aggregation"
    
    print(f"\n{'='*80}")
    print(f"Running DEFENSE Experiment: {defense_name}")
    print(f"Data: {data_dist} with {num_attackers} Attacker(s)")
    print(f"{'='*80}\n")
    
    # Backup original pyproject.toml
    original_toml = Path("pyproject.toml")
    backup_toml = Path("pyproject_backup.toml")
    
    if defense_type == "loss":
        defense_toml = Path("pyproject_defense.toml")
    else:  # median
        defense_toml = Path("pyproject_median.toml")
    
    if not defense_toml.exists():
        print("❌ Error: pyproject_defense.toml not found!")
        return False
    
    try:
        # Backup original
        print("📦 Backing up original pyproject.toml...")
        shutil.copy(original_toml, backup_toml)
        
        # Switch to defense config
        print("🔄 Switching to defense configuration...")
        shutil.copy(defense_toml, original_toml)
        
        # Build config string
        config_parts = [
            "num-server-rounds=50",
            f"use-iid={str(use_iid).lower()}",
            f'attacker-ids="{attacker_ids_str}"',
            'attack-type="label_flipping"' if num_attackers > 0 else 'attack-type="none"',
        ]
        
        # Add defense-specific config
        if defense_type == "loss":
            config_parts.append("loss-threshold-percentile=75")
        
        config_string = " ".join(config_parts)
        
        # Run experiment
        cmd = ["flwr", "run", ".", "--run-config", config_string]
        
        print(f"🚀 Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, check=True)
        
        print(f"\n✅ Defense experiment completed successfully!")
        success = True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Defense experiment failed!")
        success = False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        success = False
        
    finally:
        # Restore original config
        print("\n🔄 Restoring original configuration...")
        if backup_toml.exists():
            shutil.copy(backup_toml, original_toml)
            backup_toml.unlink()
        
    return success


def main():
    """Run defense experiments interactively or all at once."""
    
    print("\n" + "="*80)
    print("DEFENSE EXPERIMENTS RUNNER (Part 2.3)")
    print("="*80)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # Run all defense experiments
        print("\nRunning ALL defense experiments (6 total)...\n")
        
        experiments = [
            (True, 0, []),      # IID baseline
            (True, 1, [0]),     # IID 1 attacker
            (True, 2, [0, 1]),  # IID 2 attackers
            (False, 0, []),     # Non-IID baseline
            (False, 1, [0]),    # Non-IID 1 attacker
            (False, 2, [0, 1]), # Non-IID 2 attackers
        ]
        
        completed = 0
        for i, (use_iid, num_att, att_ids) in enumerate(experiments, 1):
            print(f"\n--- Experiment {i}/6 ---")
            if run_defense_experiment(use_iid, num_att, att_ids):
                completed += 1
        
        print(f"\n{'='*80}")
        print(f"Completed {completed}/{len(experiments)} defense experiments")
        print(f"{'='*80}\n")
        
    else:
        # Interactive mode - run key experiments
        print("""
Defense experiments to run for Part 2.3:

The most important experiments are with attacks + defense:
  1. IID + 2 Attackers + Defense
  2. Non-IID + 2 Attackers + Defense

Optional (for completeness):
  3. IID + 1 Attacker + Defense
  4. Non-IID + 1 Attacker + Defense

Choose an option:
  1) Run key experiments (IID + Non-IID with 2 attackers)
  2) Run all 6 experiments (includes baseline and 1 attacker)
  3) Run single experiment (IID, 2 attackers)
  4) Exit

""")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🎯 Running key experiments...")
            run_defense_experiment(use_iid=True, num_attackers=2)
            run_defense_experiment(use_iid=False, num_attackers=2)
            
        elif choice == "2":
            print("\n🎯 Running all experiments...")
            subprocess.run([sys.executable, __file__, "--all"])
            
        elif choice == "3":
            print("\n🎯 Running single experiment...")
            run_defense_experiment(use_iid=True, num_attackers=2)
            
        else:
            print("\n👋 Exiting...")
            return
    
    print("\n" + "="*80)
    print("Next step: Run 'python analyze_defense.py' to generate comparison plots")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
