"""
Quick test script to run a single baseline experiment.
Use this to verify the setup before running the full experiment suite.
"""

import subprocess
from pathlib import Path


def run_test_experiment():
    """Run a quick test experiment with baseline (no attackers)."""
    
    print("\n" + "="*80)
    print("RUNNING TEST EXPERIMENT")
    print("Configuration: FedAvg, IID data, No attackers, 10 rounds")
    print("="*80 + "\n")
    
    cmd = [
        "flwr", "run", ".",
        "--run-config",
        'num-server-rounds=10 use-iid=true strategy="FedAvg" attacker-ids="none" attack-type="none"',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=Path(__file__).parent
        )
        
        print("\n" + "="*80)
        print("✓ TEST EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("1. Check 'results/' directory for output files")
        print("2. Run 'python run_experiments.py' to run all experiments")
        print("3. Run 'python analyze_results.py' to generate analysis")
        print("="*80 + "\n")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n" + "="*80)
        print("✗ TEST EXPERIMENT FAILED!")
        print("="*80)
        print("\nPlease check:")
        print("1. Virtual environment is activated")
        print("2. All dependencies are installed (see README)")
        print("3. You're in the correct directory")
        print("="*80 + "\n")
        return False


if __name__ == "__main__":
    run_test_experiment()
