# NOTE: this file is AI generated and has not been thoroughly reviewed.
# It is only used as a sanity check for explaining differences between sb3 and r2l
#

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

SEED = 0
EVAL_FREQUENCY = 10_000
EVAL_EPISODES = 5
ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "assets" / "ppo.yaml"
LOG_DIR = ROOT_DIR / "logs" / "sb3"


def schedule(value: float | str) -> float | Callable[[float], float]:
    if isinstance(value, str) and value.startswith("lin_"):
        initial_value = float(value.removeprefix("lin_"))
        return lambda progress_remaining: progress_remaining * initial_value
    return float(value)


def normalization_options(value: bool | str) -> tuple[bool, bool]:
    if isinstance(value, bool):
        return value, value
    options = ast.literal_eval(value)
    return bool(options["norm_obs"]), bool(options["norm_reward"])


def load_config(environment: str) -> dict[str, Any]:
    with CONFIG_PATH.open() as config_file:
        configs = yaml.safe_load(config_file)
    if environment not in configs:
        raise ValueError(f"{environment} is not present in {CONFIG_PATH}")
    config = configs[environment]
    if config["policy"] != "MlpPolicy":
        raise ValueError(f"{environment} does not use MlpPolicy")
    return config


def policy_kwargs(config: dict[str, Any]) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {}
    if config["log_std_init"] != 0:
        kwargs["log_std_init"] = config["log_std_init"]
    raw_kwargs = config["policy_kwargs"]
    if isinstance(raw_kwargs, str) and "ortho_init=False" in raw_kwargs:
        kwargs["ortho_init"] = False
    return kwargs or None


def write_evaluations_csv(output_dir: Path) -> None:
    with np.load(output_dir / "evaluations.npz") as evaluations:
        with (output_dir / "evaluations.csv").open("w", newline="") as csv_file:
            writer = csv.writer(csv_file, lineterminator="\n")
            writer.writerow(("average_reward", "total_episodes"))
            for rewards in evaluations["results"]:
                writer.writerow((float(np.mean(rewards)), len(rewards)))


def train(environment: str) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if environment == "popgym-BattleshipEasy-v0":
        import popgym  # noqa: F401

    config = load_config(environment)
    norm_obs, norm_reward = normalization_options(config["normalize"])
    training_env = make_vec_env(environment, n_envs=config["n_envs"], seed=SEED)
    if norm_obs or norm_reward:
        training_env = VecNormalize(
            training_env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
            gamma=config["gamma"],
        )

    output_dir = LOG_DIR / environment
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_env = make_vec_env(environment, n_envs=1, seed=SEED + 1)
    if norm_obs or norm_reward:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=norm_obs,
            norm_reward=False,
            gamma=config["gamma"],
        )

    model = PPO(
        config["policy"],
        training_env,
        learning_rate=schedule(config["learning_rate"]),
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=schedule(config["clip_range"]),
        clip_range_vf=config["clip_range_vf"],
        normalize_advantage=config["normalize_advantage"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        use_sde=config["use_sde"],
        sde_sample_freq=config["sde_sample_freq"],
        policy_kwargs=policy_kwargs(config),
        stats_window_size=config["stats_window_size"],
        seed=SEED,
        device="cpu",
        verbose=1,
    )
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=max(EVAL_FREQUENCY // config["n_envs"], 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

    try:
        model.learn(total_timesteps=config["n_timesteps"], callback=callback)
        write_evaluations_csv(output_dir)
    finally:
        training_env.close()
        eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("environment")
    args = parser.parse_args()
    train(args.environment)


if __name__ == "__main__":
    main()
