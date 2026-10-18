# Crafter

Default training environment ([Crafter](https://github.com/danijar/crafter) survival world).

**Observation:** `uint8` rgb `(64, 64, 3)`.

**Actions:** 17 discrete skills (`noop`, movement, `do`, crafting, …). Names appear in the episodes table as `action <name>` counts.

**Episodes:** Death terminates; step limit truncates (`ENVIRONMENT__EPISODE_LENGTH`, default 1000).

**Score:** geometric mean of per-achievement success rates (percent): `exp(mean(log(1 + rate))) - 1`. Training logs cumulative score at each rollout; compare at 1M steps to published baselines (tuned PPO ~15.6, DreamerV3 ~14.8, achievement distillation ~21.8).

**Code:** [`src/noitarl/environment/crafter.py`](../src/noitarl/environment/crafter.py), score in [`crafter_score.py`](../src/noitarl/environment/crafter_score.py).
