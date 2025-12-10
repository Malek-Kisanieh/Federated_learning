================================================================================
PART 2.3: DEFENSE MECHANISM IMPLEMENTATION
================================================================================

OVERVIEW
========
This implementation adds a defense mechanism against label flipping attacks
for the federated learning system. The defense is compared against both
baseline (no attack) and attacked models.

DEFENSE MECHANISM: LOSS-BASED ANOMALY DETECTION
================================================

Concept:
--------
Malicious clients performing label flipping attacks will have significantly
higher training losses because they're training on incorrect labels. By
monitoring and filtering clients with abnormally high losses, we can identify
and exclude potentially malicious participants.

How It Works:
-------------
1. After each round of local training, collect training loss from all clients
2. Calculate the 75th percentile of all training losses
3. Identify clients with losses above this threshold
4. Exclude these clients from the model aggregation step
5. Aggregate only the models from "trusted" (low-loss) clients

Benefits:
---------
- Simple and effective for label-based attacks
- No cryptographic overhead
- Works in real-time during training
- Adaptive to different attack intensities

Limitations:
------------
- May exclude honest clients with difficult data partitions
- Sophisticated attackers might evade by limiting attack intensity
- Requires sufficient honest clients for accurate threshold

IMPLEMENTATION FILES
====================

1. flower2025/server_app_defense.py
   - Modified FedAvg strategy with defense mechanism
   - FedAvgDefense class extends FedAvg
   - Implements aggregate_fit() with loss filtering

2. analyze_defense.py
   - Comparison analysis tool
   - Generates plots comparing baseline/attacked/defended
   - Calculates defense effectiveness metrics

3. DEFENSE_INSTRUCTIONS.py
   - Step-by-step guide to run defense experiments
   - Manual commands for each scenario

RUNNING DEFENSE EXPERIMENTS
============================

For each scenario (IID/non-IID with 1 or 2 attackers), run:

flwr run . --app-dir . --serverapp flower2025.server_app_defense:app \
    --run-config 'num-server-rounds=50 use-iid=true \
    attacker-ids="0,1" attack-type="label_flipping" \
    loss-threshold-percentile=75'

This will generate:
- results/metrics_FedAvgDefense_iid_2attackers.json
- results/final_model_FedAvgDefense_iid_2attackers.pt

EXPERIMENTS NEEDED FOR PART 2.3
================================

You need to run 6 defense experiments total:

IID Data:
  1. Baseline + Defense (0 attackers) - for reference
  2. 1 Attacker + Defense
  3. 2 Attackers + Defense

Non-IID Data:
  4. Baseline + Defense (0 attackers) - for reference
  5. 1 Attacker + Defense
  6. 2 Attackers + Defense

ANALYSIS & COMPARISON
======================

After running defense experiments, use:

    python analyze_defense.py

This will generate:

1. Comparison Plots (in plots/ directory):
   - defense_comparison_iid_1attackers.png
   - defense_comparison_iid_2attackers.png
   - defense_comparison_non_iid_1attackers.png
   - defense_comparison_non_iid_2attackers.png
   
   Each plot shows 5 metrics over 50 rounds:
   - Baseline (green) - no attack, no defense
   - Attacked (red) - with attack, no defense
   - Defended (blue) - with attack, WITH defense

2. Analysis Report (defense_analysis_report.txt):
   - Quantitative comparison of all scenarios
   - Attack impact calculation
   - Defense recovery percentage
   - Conclusions and recommendations

KEY METRICS TO REPORT
======================

For your report (section 1.4), include:

1. Attack Impact:
   - Original model accuracy: X%
   - Attacked model accuracy: Y%
   - Degradation: (X-Y)%

2. Defense Effectiveness:
   - Defended model accuracy: Z%
   - Recovery: (Z-Y)% of attack mitigated
   - Remaining gap vs baseline: (X-Z)%

3. Visual Evidence:
   - Include comparison plots showing three curves
   - Highlight where defense activates (client filtering)
   - Show convergence patterns

EXAMPLE RESULTS (Expected)
===========================

Typical results for 2 attackers on IID data:

Baseline:    75% accuracy
Attacked:    56% accuracy (19% degradation)
Defended:    68% accuracy (12% recovery, 7% remaining gap)

Defense Recovery: 12/19 = 63% of attack impact mitigated

This demonstrates:
- Attack is effective (19% drop)
- Defense works (12% recovery)
- Some tradeoff remains (7% gap)

DEFENSE MECHANISM EXPLANATION
==============================

For your report, explain the defense as follows:

"The implemented defense uses loss-based anomaly detection to identify
malicious clients. During each aggregation round, the server collects
training losses from all participating clients. Clients whose losses
exceed the 75th percentile are excluded from aggregation.

This approach leverages the observation that malicious clients performing
label flipping attacks train on incorrect labels, resulting in higher
losses compared to honest clients. By filtering these outliers, the
defense maintains model integrity even when some participants are
compromised.

The defense successfully reduces attack impact from X% to Y%, recovering
Z% of the performance degradation. While some gap remains compared to
the baseline, this tradeoff is acceptable given the security benefits
and low computational overhead."

CUSTOMIZATION OPTIONS
======================

Adjust defense sensitivity by changing the threshold:

- More strict (fewer clients excluded):
  loss-threshold-percentile=85  # Only exclude top 15%

- More lenient (more clients excluded):
  loss-threshold-percentile=60  # Exclude top 40%

- Balanced (default):
  loss-threshold-percentile=75  # Exclude top 25%

ALTERNATIVE DEFENSE MECHANISMS
===============================

If time permits, you could also implement:

1. Median Aggregation
   - Use median instead of mean for weight aggregation
   - More robust to outliers

2. Krum
   - Select subset of clients closest to each other
   - Explicitly Byzantine-robust

3. FoolsGold
   - Track client contribution history
   - Detect coordinated attacks

4. Norm-Based Filtering
   - Filter updates with large gradient norms
   - Detect poisoning attacks

The loss-based approach is simpler but effective for your assignment.

TIMELINE
========

- Defense experiments: ~1 hour (6 scenarios × 10 min each)
- Analysis generation: ~2 minutes
- Total: ~1 hour

DELIVERABLES FOR SECTION 1.4
=============================

Include in your report:

□ Description of defense mechanism
□ Explanation of how it works
□ Attack impact quantification
□ Defense effectiveness quantification
□ Comparison plots (baseline vs attacked vs defended)
□ Analysis of results
□ Limitations and future improvements
□ Conclusion on FL system resilience with defense

================================================================================
