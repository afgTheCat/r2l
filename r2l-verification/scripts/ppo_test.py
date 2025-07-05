import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper, VecNormalize


def ppo_test1():
    vec_env = make_vec_env("CartPole-v1", n_envs=10)
    env = VecNormalize(vec_env)
    model = PPO("MlpPolicy", env, verbose=1)
    eval_env = VecNormalize(make_vec_env("CartPole-v1", n_envs=10))
    eval_callback = EvalCallback(
        eval_env,
        log_path=None,
        verbose=1,
        render=False,
        deterministic=True,
        eval_freq=1000,
    )
    model.learn(total_timesteps=500000, callback=eval_callback)
    print(eval_callback.evaluations_results)


def ppo_test2():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    eval_callback = EvalCallback(
        gym.make("CartPole-v1", render_mode="rgb_array"),
        log_path=None,
        verbose=1,
        render=False,
        deterministic=True,
        eval_freq=10000,
    )
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500000, callback=eval_callback)

def ppo_test3():
    vec_env = make_vec_env("CartPole-v1", n_envs=10)
    eval_callback = EvalCallback(
        gym.make("CartPole-v1", render_mode="rgb_array"),
        log_path=None,
        verbose=1,
        render=False,
        deterministic=True,
        eval_freq=1000,
    )
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=500000, callback=eval_callback)
    print(eval_callback.evaluations_results)

if __name__ == "__main__":
    ppo_test1()
    # ppo_test2()
    # ppo_test3()
