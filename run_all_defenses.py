"""
Unified Defense Experiment Runner for Part 2.3 and Part 2.4
Supports both Loss-Based Filtering and Median Aggregation defenses
"""

import subprocess
import shutil
from pathlib import Path


def backup_and_switch_pyproject(defense_type):
    """
    Backup current pyproject.toml and switch to defense-specific version.
    
    Args:
        defense_type: "loss" for Loss-Based Defense, "median" for Median Aggregation
    
    Returns:
        bool: True if switch successful
    """
    pyproject_path = Path("pyproject.toml")
    backup_path = Path("pyproject.toml.backup")
    
    if defense_type == "loss":
        defense_config = Path("pyproject_defense.toml")
    elif defense_type == "median":
        defense_config = Path("pyproject_median.toml")
    else:
        print(f"❌ Invalid defense type: {defense_type}")
        return False
    
    if not defense_config.exists():
        print(f"❌ Defense config not found: {defense_config}")
        return False
    
    # Backup current pyproject.toml
    if pyproject_path.exists():
        shutil.copy(pyproject_path, backup_path)
        print(f"✓ Backed up current pyproject.toml")
    
    # Switch to defense config
    shutil.copy(defense_config, pyproject_path)
    print(f"✓ Switched to {defense_config}")
    
    return True


def restore_pyproject():
    """Restore original pyproject.toml from backup."""
    pyproject_path = Path("pyproject.toml")
    backup_path = Path("pyproject.toml.backup")
    
    if backup_path.exists():
        shutil.copy(backup_path, pyproject_path)
        backup_path.unlink()
        print(f"✓ Restored original pyproject.toml")
    else:
        print("⚠️  No backup found to restore")


def run_defense_experiment(defense_type, data_dist, num_attackers, loss_threshold=75):
    """
    Run a single defense experiment.
    
    Args:
        defense_type: "loss" or "median"
        data_dist: "iid" or "non_iid"
        num_attackers: 0, 1, or 2
        loss_threshold: Percentile threshold for loss-based defense (default: 75)
    """
    
    print(f"\n{'='*80}")
    print(f"Running {defense_type.upper()} Defense: {data_dist} with {num_attackers} attacker(s)")
    print(f"{'='*80}\n")
    
    # Switch to appropriate pyproject config
    if not backup_and_switch_pyproject(defense_type):
        return False
    
    try:
        # Build attacker-ids string
        if num_attackers == 0:
            attacker_ids = "none"
        elif num_attackers == 1:
            attacker_ids = "3"
        elif num_attackers == 2:
            attacker_ids = "3,4"
        else:
            print(f"❌ Invalid number of attackers: {num_attackers}")
            return False
        
        # Build config string based on defense type
        if defense_type == "loss":
            config_str = f'data-distribution="{data_dist}" attacker-ids="{attacker_ids}" loss-threshold-percentile={loss_threshold}'
        else:  # median
            config_str = f'data-distribution="{data_dist}" attacker-ids="{attacker_ids}"'
        
        # Build and run flwr command
        cmd = [
            "flower-simulation",
            "--app", ".",
            "--run-config", config_str,
            "--num-supernodes", "5"
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        print(f"\n✓ Experiment completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment failed with error code {e.returncode}")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    
    finally:
        # Always restore original pyproject.toml
        restore_pyproject()


def run_all_loss_defense():
    """Run all Loss-Based Defense experiments."""
    print("\n" + "="*80)
    print("RUNNING ALL LOSS-BASED DEFENSE EXPERIMENTS (Part 2.3)")
    print("="*80)
    
    experiments = [
        ("iid", 1),
        ("iid", 2),
        ("non_iid", 1),
        ("non_iid", 2),
    ]
    
    success_count = 0
    for data_dist, num_attackers in experiments:
        if run_defense_experiment("loss", data_dist, num_attackers):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Loss-Based Defense: {success_count}/{len(experiments)} experiments completed")
    print(f"{'='*80}\n")


def run_all_median_defense():
    """Run all Median Aggregation Defense experiments."""
    print("\n" + "="*80)
    print("RUNNING ALL MEDIAN AGGREGATION DEFENSE EXPERIMENTS (Part 2.4)")
    print("="*80)
    
    experiments = [
        ("iid", 1),
        ("iid", 2),
        ("non_iid", 1),
        ("non_iid", 2),
    ]
    
    success_count = 0
    for data_dist, num_attackers in experiments:
        if run_defense_experiment("median", data_dist, num_attackers):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"Median Defense: {success_count}/{len(experiments)} experiments completed")
    print(f"{'='*80}\n")


def run_all_defenses():
    """Run all defense experiments for both mechanisms."""
    print("\n" + "="*80)
    print("RUNNING ALL DEFENSE EXPERIMENTS (Parts 2.3 & 2.4)")
    print("="*80)
    
    print("\n[1/2] Running Loss-Based Defense experiments...")
    run_all_loss_defense()
    
    print("\n[2/2] Running Median Aggregation Defense experiments...")
    run_all_median_defense()
    
    print("\n" + "="*80)
    print("ALL DEFENSE EXPERIMENTS COMPLETED!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run: python analyze_defense.py (for Part 2.3 analysis)")
    print("  2. Run: python compare_defenses.py (for Part 2.4 comparison)")
    print("="*80 + "\n")


def main():
    """Main entry point with interactive menu."""
    
    while True:
        print("\n" + "="*80)
        print("FEDERATED LEARNING DEFENSE EXPERIMENTS")
        print("="*80)
        print("\n1. Run all Loss-Based Defense experiments (Part 2.3)")
        print("2. Run all Median Aggregation Defense experiments (Part 2.4)")
        print("3. Run ALL defense experiments (both mechanisms)")
        print("4. Run single Loss-Based Defense experiment")
        print("5. Run single Median Aggregation experiment")
        print("6. Exit")
        print("\n" + "="*80)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            run_all_loss_defense()
        
        elif choice == "2":
            run_all_median_defense()
        
        elif choice == "3":
            run_all_defenses()
        
        elif choice == "4":
            print("\nLoss-Based Defense Experiment")
            print("-" * 40)
            data_dist = input("Data distribution (iid/non_iid): ").strip().lower()
            if data_dist not in ["iid", "non_iid"]:
                print("❌ Invalid data distribution!")
                continue
            
            try:
                num_attackers = int(input("Number of attackers (0/1/2): ").strip())
                if num_attackers not in [0, 1, 2]:
                    print("❌ Invalid number of attackers!")
                    continue
            except ValueError:
                print("❌ Invalid number!")
                continue
            
            try:
                threshold = int(input("Loss threshold percentile (default 75): ").strip() or "75")
            except ValueError:
                threshold = 75
            
            run_defense_experiment("loss", data_dist, num_attackers, threshold)
        
        elif choice == "5":
            print("\nMedian Aggregation Defense Experiment")
            print("-" * 40)
            data_dist = input("Data distribution (iid/non_iid): ").strip().lower()
            if data_dist not in ["iid", "non_iid"]:
                print("❌ Invalid data distribution!")
                continue
            
            try:
                num_attackers = int(input("Number of attackers (0/1/2): ").strip())
                if num_attackers not in [0, 1, 2]:
                    print("❌ Invalid number of attackers!")
                    continue
            except ValueError:
                print("❌ Invalid number!")
                continue
            
            run_defense_experiment("median", data_dist, num_attackers)
        
        elif choice == "6":
            print("\nExiting...")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-6.")


if __name__ == "__main__":
    main()
