# riftgym

A Python RL environment and training library for the [brokenwings](https://github.com/MiscellaneousStuff/brokenwings) game server (a 1.0.0.126 League of Legends reimplementation). Modeled in spirit on DeepMind's [pysc2](https://github.com/google-deepmind/pysc2).

## Install

```bash
pip install -e .              # core env + bridge (gymnasium + numpy)
pip install -e ".[sb3]"       # adds stable-baselines3 for the eval/train CLIs
pip install -e ".[dev]"       # ruff + pyright + pytest
```

Python 3.11+ required. Docker required for the container deployment.

## Quickstart — eval a checkpoint vs the engine BT

The brokenwings image isn't published to a registry yet, so build it locally first (one-off, ~30s after first build):

```bash
cd ../brokenwings && docker compose -f docker/compose.yaml build server
```

Drop a trained MaskablePPO checkpoint at `checkpoints/agent.zip`, then:

```bash
riftgym-eval ./checkpoints/agent.zip --vs-engine-bot --episodes 50 \
    --image brokenwings --pull never
```

This pulls the agent in `cid=0`, leaves `cid=1` engine-controlled (the in-tree Ezreal behaviour tree), tightens the spawn to ~849u apart so the BT engages on respawn, and prints W/L/D + win rate per episode. The brokenwings 40k-step baseline scores ~77% over 100 episodes against the BT.

Once the GHCR release workflow lands the public image, `--image` and `--pull` will be optional.

## Library usage

### Connect to an already-running server

```python
from riftgym.env.lol_gym import LoLGymEnv

env = LoLGymEnv(host="127.0.0.1", port=5120)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

### Auto-launch a containerized server

```python
from riftgym.launcher import ServerLauncher
from riftgym.run_configs import ContainerRunConfig
from riftgym.env.lol_gym import LoLGymEnv

rc = ContainerRunConfig(
    image="brokenwings",
    tag="latest",
    pull_policy="never",
    entrypoint=("/usr/bin/tini", "--", "/app/entrypoints/server.sh"),
)
with ServerLauncher(n=1, run_config=rc) as launcher:
    env = LoLGymEnv(port=launcher.rl_ports[0], claim_opp=False, omit_opp_action=True)
    # ... drive env ...
    env.close()
```

`ContainerHandle` registers an `atexit` hook on construction, so a Ctrl+C out of an eval still tears down the container and frees the host ports.

### Custom opponent policy

```python
def my_opp(env, obs):
    return {"type": "move", "client_id": env.opp_cid, "x": 7000.0, "y": 7000.0}

env = LoLGymEnv(port=5120, opp_policy=my_opp)
```

The default `opp_policy` is uniform-random (seeded via `reset(seed=...)`).

## Action / observation space

- **Action space** — `Discrete(13)`:
  - 0..7  move 300u in 8 compass dirs (N, NE, E, SE, S, SW, W, NW)
  - 8     attack opponent (target net_id)
  - 9..12 cast Q / W / E / R aimed at opponent's current position
- **Observation space** — `Box(float32, (110,))`. Per champ (me then opp): 7 base feats + 8 nearest missiles × 6 fields each. See [`riftgym/lib/encoding.py`](src/riftgym/lib/encoding.py) for exact normalization.
- **Action mask** — `env.action_masks()` returns the 13-bool mask used by sb3-contrib's `MaskablePPO`. Spells are masked off when on cooldown or mana-blocked.

## Roadmap

- [x] [#1](https://github.com/MiscellaneousStuff/riftgym/issues/1) Containerized eval vs engine BT
- [ ] Multilane envs (parallel 1v1s per server) and `make_multilane_envs` helper
- [ ] PPO trainer port (`riftgym-train`, `ThreadVecEnv`, `SnapshotPool`)
- [ ] GHCR image publish workflow on the brokenwings side
- [ ] Optional shared-secret bridge auth (cross-host setups; v0.2 of the bridge protocol)
- [ ] PettingZoo multi-agent API for true N-agent training

## License

MIT — see [LICENSE](LICENSE).
