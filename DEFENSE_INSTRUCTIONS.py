"""
Simple script to run defense experiments by manually calling flwr with defense server app.
For Part 2.3: Defense Implementation

Run this script to see instructions: python DEFENSE_INSTRUCTIONS.py
"""

import os

print("""
================================================================================
PART 2.3: DEFENSE EXPERIMENTS
================================================================================

Defense Mechanism: Loss-Based Anomaly Detection
- Filters out clients with training losses above the 75th percentile
- Detects and excludes potentially malicious clients during aggregation
- Maintains model performance under attack

To run defense experiments, use these commands:
================================================================================

# IID Data - Baseline with Defense
flwr run . --app-dir . --serverapp flower2025.server_app_defense:app --run-config 'num-server-rounds=50 use-iid=true attacker-ids="none" attack-type="none" loss-threshold-percentile=75'

# IID Data - 1 Attacker with Defense
flwr run . --app-dir . --serverapp flower2025.server_app_defense:app --run-config 'num-server-rounds=50 use-iid=true attacker-ids="0" attack-type="label_flipping" loss-threshold-percentile=75'

# IID Data - 2 Attackers with Defense
flwr run . --app-dir . --serverapp flower2025.server_app_defense:app --run-config 'num-server-rounds=50 use-iid=true attacker-ids="0,1" attack-type="label_flipping" loss-threshold-percentile=75'

# Non-IID Data - Baseline with Defense  
flwr run . --app-dir . --serverapp flower2025.server_app_defense:app --run-config 'num-server-rounds=50 use-iid=false attacker-ids="none" attack-type="none" loss-threshold-percentile=75'

# Non-IID Data - 1 Attacker with Defense
flwr run . --app-dir . --serverapp flower2025.server_app_defense:app --run-config 'num-server-rounds=50 use-iid=false attacker-ids="0" attack-type="label_flipping" loss-threshold-percentile=75'

# Non-IID Data - 2 Attackers with Defense
flwr run . --app-dir . --serverapp flower2025.server_app_defense:app --run-config 'num-server-rounds=50 use-iid=false attacker-ids="0,1" attack-type="label_flipping" loss-threshold-percentile=75'

================================================================================

Results will be saved as:
- results/metrics_FedAvgDefense_iid_0attackers.json
- results/metrics_FedAvgDefense_iid_1attackers.json
- results/metrics_FedAvgDefense_iid_2attackers.json
- results/metrics_FedAvgDefense_non_iid_0attackers.json
- results/metrics_FedAvgDefense_non_iid_1attackers.json
- results/metrics_FedAvgDefense_non_iid_2attackers.json

After running, use 'python analyze_defense.py' to compare:
- Baseline (no attack, no defense)
- Attacked (with attack, no defense) 
- Defended (with attack, with defense)

================================================================================
""")
