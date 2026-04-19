# NOTE: this file is AI generated has not been throughly reviewed. 
# It is only used as a sanity check for explaining differences between sb3 and r2l

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize


@dataclass(frozen=True)
class PPOConfig:
    env_name: str
    n_envs: int
    clip_range: float | None
    ent_coef: float
    gae_lambda: float
    gamma: float
    learning_rate: float | None
    batch_size: int | None
    n_epochs: int
    n_steps: int
    total_timesteps: int
    vf_coef: float | None
    max_grad_norm: float | None
    use_sde: bool | None
    sde_sample_freq: int | None


CONFIGS: dict[str, PPOConfig] = {
    "CartPole-v1": PPOConfig(
        env_name="CartPole-v1",
        n_envs=8,
        clip_range=0.2,
        ent_coef=0.0,
        gae_lambda=0.8,
        gamma=0.98,
        learning_rate=0.001,
        batch_size=256,
        n_epochs=20,
        n_steps=32,
        total_timesteps=100_000,
        vf_coef=None,
        max_grad_norm=None,
        use_sde=None,
        sde_sample_freq=None,
    ),
    "Pendulum-v1": PPOConfig(
        env_name="Pendulum-v1",
        n_envs=4,
        clip_range=0.2,
        ent_coef=0.0,
        gae_lambda=0.95,
        gamma=0.9,
        learning_rate=0.001,
        batch_size=None,
        n_epochs=10,
        n_steps=1024,
        total_timesteps=100_000,
        vf_coef=None,
        max_grad_norm=None,
        use_sde=True,
        sde_sample_freq=4,
    ),
    "Acrobot-v1": PPOConfig(
        env_name="Acrobot-v1",
        n_envs=16,
        clip_range=None,
        ent_coef=0.0,
        gae_lambda=0.94,
        gamma=0.99,
        learning_rate=None,
        batch_size=None,
        n_epochs=4,
        n_steps=256,
        total_timesteps=1_000_000,
        vf_coef=None,
        max_grad_norm=None,
        use_sde=None,
        sde_sample_freq=None,
    ),
    "MountainCar-v0": PPOConfig(
        env_name="MountainCar-v0",
        n_envs=16,
        clip_range=None,
        ent_coef=0.0,
        gae_lambda=0.98,
        gamma=0.99,
        learning_rate=None,
        batch_size=None,
        n_epochs=4,
        n_steps=16,
        total_timesteps=1_000_000,
        vf_coef=None,
        max_grad_norm=None,
        use_sde=None,
        sde_sample_freq=None,
    ),
    "MountainCarContinuous-v0": PPOConfig(
        env_name="MountainCarContinuous-v0",
        n_envs=1,
        clip_range=0.1,
        ent_coef=0.00429,
        gae_lambda=0.9,
        gamma=0.9999,
        learning_rate=7.77e-05,
        batch_size=256,
        n_epochs=10,
        n_steps=8,
        total_timesteps=20_000,
        vf_coef=0.19,
        max_grad_norm=5.0,
        use_sde=True,
        sde_sample_freq=None,
    ),
    "LunarLander-v2": PPOConfig(
        env_name="LunarLander-v2",
        n_envs=16,
        clip_range=None,
        ent_coef=0.01,
        gae_lambda=0.98,
        gamma=0.999,
        learning_rate=None,
        batch_size=64,
        n_epochs=4,
        n_steps=1024,
        total_timesteps=1_000_000,
        vf_coef=None,
        max_grad_norm=None,
        use_sde=None,
        sde_sample_freq=None,
    ),
    "LunarLanderContinuous-v2": PPOConfig(
        env_name="LunarLanderContinuous-v2",
        n_envs=16,
        clip_range=None,
        ent_coef=0.01,
        gae_lambda=0.98,
        gamma=0.999,
        learning_rate=None,
        batch_size=64,
        n_epochs=4,
        n_steps=1024,
        total_timesteps=1_000_000,
        vf_coef=None,
        max_grad_norm=None,
        use_sde=None,
        sde_sample_freq=None,
    ),
}


class RewardLogger(BaseCallback):
    def __init__(self, log_every_rollouts: int) -> None:
        super().__init__()
        self.log_every_rollouts = log_every_rollouts
        self.rollout_idx = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_idx += 1
        if self.rollout_idx % self.log_every_rollouts != 0:
            return
        ep_info_buffer = self.model.ep_info_buffer
        if not ep_info_buffer:
            print(f"rollout={self.rollout_idx} reward=n/a")
            return
        mean_reward = sum(info["r"] for info in ep_info_buffer) / len(ep_info_buffer)
        print(f"rollout={self.rollout_idx} reward={mean_reward:.3f}")


def build_model(
    config: PPOConfig,
    seed: int,
    norm_obs: bool = False,
    norm_reward: bool = False,
) -> PPO:
    env = make_vec_env(config.env_name, n_envs=config.n_envs, seed=seed)
    if norm_obs or norm_reward:
        env = VecNormalize(env, norm_obs=norm_obs, norm_reward=norm_reward)

    kwargs: dict[str, object] = {
        "env": env,
        "policy": "MlpPolicy",
        "n_steps": config.n_steps,
        "batch_size": config.batch_size or 64,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "ent_coef": config.ent_coef,
        "verbose": 1,
        "seed": seed,
    }
    if config.clip_range is not None:
        kwargs["clip_range"] = config.clip_range
    if config.learning_rate is not None:
        kwargs["learning_rate"] = config.learning_rate
    if config.vf_coef is not None:
        kwargs["vf_coef"] = config.vf_coef
    if config.max_grad_norm is not None:
        kwargs["max_grad_norm"] = config.max_grad_norm
    if config.use_sde is not None:
        kwargs["use_sde"] = config.use_sde
    if config.sde_sample_freq is not None:
        kwargs["sde_sample_freq"] = config.sde_sample_freq
    return PPO(**kwargs)


def train(
    config: PPOConfig,
    seed: int,
    eval_freq: int,
    eval_episodes: int,
    norm_obs: bool = False,
    norm_reward: bool = False,
) -> None:
    model = build_model(config, seed, norm_obs=norm_obs, norm_reward=norm_reward)

    eval_env = make_vec_env(config.env_name, n_envs=1, seed=seed + 1)
    if norm_obs or norm_reward:
        eval_env = VecNormalize(
            eval_env,
            training=False,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
        )
        eval_env.obs_rms = model.get_env().obs_rms
        eval_env.ret_rms = model.get_env().ret_rms
    log_dir = Path(__file__).resolve().parent / "runs" / config.env_name
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        RewardLogger(log_every_rollouts=10),
        EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir),
            log_path=str(log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        ),
    ]

    print(
        f"training with VecNormalize norm_obs={norm_obs} norm_reward={norm_reward}"
    )
    print(config)
    model.learn(total_timesteps=config.total_timesteps, callback=callbacks)


def configure_ppo_test(
    config: PPOConfig,
    norm_obs: bool = False,
    norm_reward: bool = False,
) -> None:
    train(
        config,
        seed=0,
        eval_freq=10_000,
        eval_episodes=10,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
    )


def cartpole() -> None:
    configure_ppo_test(CONFIGS["CartPole-v1"])


def pendulum() -> None:
    configure_ppo_test(CONFIGS["Pendulum-v1"])


def acrobot() -> None:
    configure_ppo_test(CONFIGS["Acrobot-v1"])


def mountain_car() -> None:
    configure_ppo_test(CONFIGS["MountainCar-v0"])


def mountain_car_norm_obs() -> None:
    configure_ppo_test(CONFIGS["MountainCar-v0"], norm_obs=True)


def mountain_car_norm_obs_reward() -> None:
    configure_ppo_test(CONFIGS["MountainCar-v0"], norm_obs=True, norm_reward=True)


def mountain_car_continuous() -> None:
    configure_ppo_test(CONFIGS["MountainCarContinuous-v0"])


def lunar_lander() -> None:
    configure_ppo_test(CONFIGS["LunarLander-v2"])


def lunar_lander_continuous() -> None:
    configure_ppo_test(CONFIGS["LunarLanderContinuous-v2"])


def main() -> None:
    mountain_car_norm_obs_reward()


if __name__ == "__main__":
    main()
