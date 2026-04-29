# riftgym

A Python RL environment and training library for the [brokenwings](https://github.com/MiscellaneousStuff/brokenwings) game server (a 1.0.0.126 League of Legends reimplementation). Modeled in spirit on DeepMind's [pysc2](https://github.com/google-deepmind/pysc2).

> Pre-alpha. The API is unstable until v0.1.

## Status

riftgym is the standalone successor to `brokenwings/tools/`. The first milestone is end-to-end eval against the engine BT through a containerized server — see [issue #1](https://github.com/MiscellaneousStuff/riftgym/issues/1).

## Install

```bash
pip install -e .              # core env + bridge
pip install -e ".[sb3]"       # adds stable-baselines3 for training/eval CLIs
pip install -e ".[dev]"       # ruff + pyright + pytest
```

Python 3.11+ required.

## Quickstart

The default deployment is container-first: riftgym pulls (or builds) the brokenwings Docker image and runs the server inside it.

```python
from riftgym import LoLGymEnv, ServerLauncher
from riftgym.run_configs import ContainerRunConfig

rc = ContainerRunConfig(image="ghcr.io/miscellaneousstuff/brokenwings", tag="latest")
with ServerLauncher(n=1, run_config=rc) as launcher:
    env = LoLGymEnv(host="127.0.0.1", port=launcher.rl_ports[0])
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
```

For brokenwings developers, swap `ContainerRunConfig` for `BrokenwingsLocalBuildRunConfig(project_root=...)` to run from a local build instead.

## License

MIT — see [LICENSE](LICENSE).
