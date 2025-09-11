import gymnasium as gym
from stable_baselines3 import A2C

def a2c_test():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)


if __name__ == "__main__":
    a2c_test()
