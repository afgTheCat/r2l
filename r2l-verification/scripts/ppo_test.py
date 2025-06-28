from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

def ppo_test():
    vec_env = make_vec_env("CartPole-v1", n_envs=16)
    env = VecNormalize(vec_env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500000)


if __name__ == "__main__":
    ppo_test()
