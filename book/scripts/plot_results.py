#!/usr/bin/env python3
"""Generate the per-environment learning curves used by the results chapter."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
CONFIG = ROOT / "benchmark" / "assets" / "ppo.yaml"
GALLERY_OUTPUT = ROOT / "book" / "src" / "images" / "results"

FRAMEWORKS = {
    "candle": ("Candle", "#E68619"),
    "burn": ("Burn", "#7B61A8"),
    "sb3": ("Stable Baselines3", "#3B82B8"),
}

ENVIRONMENTS = (
    "Acrobot-v1",
    "Ant-v4",
    "AntBulletEnv-v0",
    "BipedalWalker-v3",
    "BipedalWalkerHardcore-v3",
    "CartPole-v1",
    "HalfCheetah-v4",
    "HalfCheetahBulletEnv-v0",
    "Hopper-v4",
    "HopperBulletEnv-v0",
    "Humanoid-v4",
    "HumanoidBulletEnv-v0",
    "HumanoidStandup-v2",
    "InvertedDoublePendulum-v2",
    "InvertedDoublePendulumBulletEnv-v0",
    "InvertedPendulum-v2",
    "InvertedPendulumSwingupBulletEnv-v0",
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "Reacher-v2",
    "ReacherBulletEnv-v0",
    "Swimmer-v4",
    "Walker2DBulletEnv-v0",
    "Walker2d-v4",
    "popgym-BattleshipEasy-v0",
)


def load_rewards(path: Path) -> np.ndarray:
    with path.open(newline="") as file:
        rows = csv.DictReader(file)
        return np.asarray([float(row["average_reward"]) for row in rows])


def moving_average(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = min(51, max(3, 2 * (len(values) // 100) + 1))
    if len(values) < window:
        return np.arange(len(values)), values
    weights = np.ones(window) / window
    return np.arange(window // 2, len(values) - window // 2), np.convolve(
        values, weights, mode="valid"
    )


def sampled_steps(
    framework: str, environment: str, count: int, config: dict[str, dict]
) -> np.ndarray:
    if framework == "sb3":
        with np.load(RUNS / framework / environment / "evaluations.npz") as data:
            return np.asarray(data["timesteps"][:count])

    environment_config = config[environment]
    steps_per_rollout = environment_config["n_envs"] * environment_config["n_steps"]
    return np.arange(1, count + 1) * steps_per_rollout


def plot_environment(axis, environment: str, config: dict[str, dict]):
    handles = {}
    for framework, (label, color) in FRAMEWORKS.items():
        rewards = load_rewards(RUNS / framework / environment / "evaluations.csv")
        steps = sampled_steps(framework, environment, len(rewards), config)
        smooth_indices, smoothed_rewards = moving_average(rewards)

        axis.plot(steps, rewards, color=color, alpha=0.13, linewidth=0.8)
        (line,) = axis.plot(
            steps[smooth_indices], smoothed_rewards, color=color, linewidth=2.0
        )
        handles[label] = line

    axis.set_title(environment, fontweight="bold")
    axis.set_xlabel("Sampled environment steps")
    axis.set_ylabel("Average evaluation reward")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.grid(alpha=0.2)
    axis.margins(x=0.01)
    return handles


def main() -> None:
    with CONFIG.open() as file:
        config = yaml.safe_load(file)

    GALLERY_OUTPUT.mkdir(parents=True, exist_ok=True)
    for environment in ENVIRONMENTS:
        figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        handles = plot_environment(axis, environment, config)
        axis.legend(handles.values(), handles.keys(), frameon=False)
        output = GALLERY_OUTPUT / f"{environment}.png"
        figure.savefig(output, dpi=140, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
