# auditenv baseline report

Closed-mode evaluation on 20 episodes (10 train + 10 dev: 5 clean, 5 with interacting errors, 10 mixed).
Return = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1). P/R/F1 are micro-averaged over episodes;
`F1 mean` is the per-episode average. The experimental partial-credit F1 (type + ids only, no evidence)
is reported separately and is NOT part of the reward.

## all

| policy | P | R | F1 | F1 mean | return | steps | time s (mean/max) | exp. partial F1 |
|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 | 0.050 | -0.901 | 17.2 | 0.044/0.155 | 0.050 |
| rule_based | 1.000 | 0.657 | 0.793 | 0.801 | +0.565 | 20.1 | 0.053/0.181 | 0.801 |
| flag_all | 0.002 | 0.014 | 0.003 | 0.003 | -0.994 | 3.7 | 0.048/0.180 | 0.006 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | +1.000 | 1.0 | 0.048/0.225 | 1.000 |

## clean

| policy | P | R | F1 | F1 mean | return | steps | time s (mean/max) | exp. partial F1 |
|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 | 0.200 | -0.602 | 20.6 | 0.055/0.124 | 0.200 |
| rule_based | 0.000 | 0.000 | 1.000 | 1.000 | +0.956 | 23.0 | 0.066/0.144 | 1.000 |
| flag_all | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 3.4 | 0.057/0.146 | 0.000 |
| oracle | 0.000 | 0.000 | 1.000 | 1.000 | +1.000 | 1.0 | 0.053/0.125 | 1.000 |

## interaction

| policy | P | R | F1 | F1 mean | return | steps | time s (mean/max) | exp. partial F1 |
|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 19.0 | 0.041/0.045 | 0.000 |
| rule_based | 1.000 | 0.560 | 0.718 | 0.667 | +0.296 | 19.4 | 0.048/0.053 | 0.667 |
| flag_all | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 3.4 | 0.042/0.051 | 0.000 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | +1.000 | 1.0 | 0.039/0.052 | 1.000 |

## mixed

| policy | P | R | F1 | F1 mean | return | steps | time s (mean/max) | exp. partial F1 |
|---|---|---|---|---|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 | 0.000 | -1.000 | 14.7 | 0.041/0.155 | 0.000 |
| rule_based | 1.000 | 0.708 | 0.829 | 0.768 | +0.504 | 18.9 | 0.049/0.181 | 0.768 |
| flag_all | 0.003 | 0.021 | 0.006 | 0.006 | -0.988 | 4.0 | 0.045/0.180 | 0.012 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | +1.000 | 1.0 | 0.050/0.225 | 1.000 |

## Per-episode results

| policy | episode | category | TP | FP | FN | F1 | return | steps |
|---|---|---|---|---|---|---|---|---|
| random | train-01 | clean | 0 | 0 | 0 | 1.000 | +0.988 | 7 |
| random | train-02 | clean | 0 | 2 | 0 | 0.000 | -1.000 | 30 |
| random | train-03 | clean | 0 | 1 | 0 | 0.000 | -1.000 | 26 |
| random | train-04 | interaction | 0 | 2 | 2 | 0.000 | -1.000 | 23 |
| random | train-05 | interaction | 0 | 0 | 4 | 0.000 | -1.000 | 19 |
| random | train-06 | mixed | 0 | 0 | 4 | 0.000 | -1.000 | 16 |
| random | train-07 | mixed | 0 | 0 | 10 | 0.000 | -1.000 | 17 |
| random | train-08 | mixed | 0 | 0 | 2 | 0.000 | -1.000 | 10 |
| random | train-09 | mixed | 0 | 2 | 4 | 0.000 | -1.000 | 4 |
| random | train-10 | mixed | 0 | 0 | 6 | 0.000 | -1.000 | 13 |
| random | dev-01 | clean | 0 | 1 | 0 | 0.000 | -1.000 | 28 |
| random | dev-02 | clean | 0 | 3 | 0 | 0.000 | -1.000 | 12 |
| random | dev-03 | interaction | 0 | 0 | 4 | 0.000 | -1.000 | 26 |
| random | dev-04 | interaction | 0 | 2 | 6 | 0.000 | -1.000 | 19 |
| random | dev-05 | interaction | 0 | 0 | 9 | 0.000 | -1.000 | 8 |
| random | dev-06 | mixed | 0 | 2 | 4 | 0.000 | -1.000 | 14 |
| random | dev-07 | mixed | 0 | 0 | 8 | 0.000 | -1.000 | 21 |
| random | dev-08 | mixed | 0 | 0 | 3 | 0.000 | -1.000 | 5 |
| random | dev-09 | mixed | 0 | 0 | 6 | 0.000 | -1.000 | 25 |
| random | dev-10 | mixed | 0 | 3 | 1 | 0.000 | -1.000 | 22 |
| rule_based | train-01 | clean | 0 | 0 | 0 | 1.000 | +0.970 | 16 |
| rule_based | train-02 | clean | 0 | 0 | 0 | 1.000 | +0.952 | 25 |
| rule_based | train-03 | clean | 0 | 0 | 0 | 1.000 | +0.980 | 11 |
| rule_based | train-04 | interaction | 1 | 0 | 1 | 0.667 | +0.299 | 18 |
| rule_based | train-05 | interaction | 1 | 0 | 3 | 0.400 | -0.240 | 21 |
| rule_based | train-06 | mixed | 3 | 0 | 1 | 0.857 | +0.684 | 16 |
| rule_based | train-07 | mixed | 7 | 0 | 3 | 0.824 | +0.615 | 17 |
| rule_based | train-08 | mixed | 2 | 0 | 0 | 1.000 | +0.964 | 19 |
| rule_based | train-09 | mixed | 3 | 0 | 1 | 0.857 | +0.694 | 11 |
| rule_based | train-10 | mixed | 4 | 0 | 2 | 0.800 | +0.564 | 19 |
| rule_based | dev-01 | clean | 0 | 0 | 0 | 1.000 | +0.962 | 20 |
| rule_based | dev-02 | clean | 0 | 0 | 0 | 1.000 | +0.916 | 43 |
| rule_based | dev-03 | interaction | 2 | 0 | 2 | 0.667 | +0.299 | 18 |
| rule_based | dev-04 | interaction | 4 | 0 | 2 | 0.800 | +0.562 | 20 |
| rule_based | dev-05 | interaction | 6 | 0 | 3 | 0.800 | +0.562 | 20 |
| rule_based | dev-06 | mixed | 2 | 0 | 2 | 0.667 | +0.313 | 11 |
| rule_based | dev-07 | mixed | 5 | 0 | 3 | 0.769 | +0.440 | 50 |
| rule_based | dev-08 | mixed | 3 | 0 | 0 | 1.000 | +0.978 | 12 |
| rule_based | dev-09 | mixed | 5 | 0 | 1 | 0.909 | +0.784 | 18 |
| rule_based | dev-10 | mixed | 0 | 0 | 1 | 0.000 | -1.000 | 16 |
| flag_all | train-01 | clean | 0 | 30 | 0 | 0.000 | -1.000 | 3 |
| flag_all | train-02 | clean | 0 | 30 | 0 | 0.000 | -1.000 | 3 |
| flag_all | train-03 | clean | 0 | 30 | 0 | 0.000 | -1.000 | 3 |
| flag_all | train-04 | interaction | 0 | 30 | 2 | 0.000 | -1.000 | 3 |
| flag_all | train-05 | interaction | 0 | 30 | 4 | 0.000 | -1.000 | 3 |
| flag_all | train-06 | mixed | 0 | 30 | 4 | 0.000 | -1.000 | 5 |
| flag_all | train-07 | mixed | 0 | 30 | 10 | 0.000 | -1.000 | 5 |
| flag_all | train-08 | mixed | 0 | 30 | 2 | 0.000 | -1.000 | 4 |
| flag_all | train-09 | mixed | 0 | 30 | 4 | 0.000 | -1.000 | 3 |
| flag_all | train-10 | mixed | 0 | 30 | 6 | 0.000 | -1.000 | 4 |
| flag_all | dev-01 | clean | 0 | 30 | 0 | 0.000 | -1.000 | 3 |
| flag_all | dev-02 | clean | 0 | 30 | 0 | 0.000 | -1.000 | 5 |
| flag_all | dev-03 | interaction | 0 | 30 | 4 | 0.000 | -1.000 | 3 |
| flag_all | dev-04 | interaction | 0 | 30 | 6 | 0.000 | -1.000 | 4 |
| flag_all | dev-05 | interaction | 0 | 30 | 9 | 0.000 | -1.000 | 4 |
| flag_all | dev-06 | mixed | 0 | 30 | 4 | 0.000 | -1.000 | 3 |
| flag_all | dev-07 | mixed | 0 | 30 | 8 | 0.000 | -1.000 | 6 |
| flag_all | dev-08 | mixed | 1 | 29 | 2 | 0.061 | -0.883 | 3 |
| flag_all | dev-09 | mixed | 0 | 30 | 6 | 0.000 | -1.000 | 3 |
| flag_all | dev-10 | mixed | 0 | 30 | 1 | 0.000 | -1.000 | 4 |
| oracle | train-01 | clean | 0 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-02 | clean | 0 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-03 | clean | 0 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-04 | interaction | 2 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-05 | interaction | 4 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-06 | mixed | 4 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-07 | mixed | 10 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-08 | mixed | 2 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-09 | mixed | 4 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | train-10 | mixed | 6 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-01 | clean | 0 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-02 | clean | 0 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-03 | interaction | 4 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-04 | interaction | 6 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-05 | interaction | 9 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-06 | mixed | 4 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-07 | mixed | 8 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-08 | mixed | 3 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-09 | mixed | 6 | 0 | 0 | 1.000 | +1.000 | 1 |
| oracle | dev-10 | mixed | 1 | 0 | 0 | 1.000 | +1.000 | 1 |

## Paired checks: 20/20 passed

| episode | transform | validator | oracle reward equal | rule-based reward equal | rule-based answer equal |
|---|---|---|---|---|---|
| train-04 | permute | True | True | True | True |
| train-04 | remap_ids | True | True | True | True |
| train-05 | permute | True | True | True | True |
| train-05 | remap_ids | True | True | True | True |
| dev-03 | permute | True | True | True | True |
| dev-03 | remap_ids | True | True | True | True |
| dev-04 | permute | True | True | True | True |
| dev-04 | remap_ids | True | True | True | True |
| dev-05 | permute | True | True | True | True |
| dev-05 | remap_ids | True | True | True | True |
| train-06 | permute | True | True | True | True |
| train-06 | remap_ids | True | True | True | True |
| train-07 | permute | True | True | True | True |
| train-07 | remap_ids | True | True | True | True |
| train-08 | permute | True | True | True | True |
| train-08 | remap_ids | True | True | True | True |
| train-09 | permute | True | True | True | True |
| train-09 | remap_ids | True | True | True | True |
| train-10 | permute | True | True | True | True |
| train-10 | remap_ids | True | True | True | True |
