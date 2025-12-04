# ppo_cmpr_study


I tested three PPO modifications. Baseline PPO scored ≈185 with high variance (std ≈48). Adding observation normalization improved both stability and performance (≈199, std ≈18). In contrast, distance-based reward shaping reduced the mean to ≈160, and stronger exploration (gSDE/param-noise) performed even worse (≈84–147). So, while obs-norm is clearly beneficial, extra shaping and exploration actually degraded PPO on this already well-tuned environment.
