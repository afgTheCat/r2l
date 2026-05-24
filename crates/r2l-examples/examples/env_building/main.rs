// ANCHOR: env_builders
use anyhow::{Ok, Result};
use r2l_api::{Env, EnvBuilder, EnvDescription, PPO2AlgorithmBuilder, Snapshot, Space, TensorData};
use r2l_gym::GymEnvBuilder;

// Not a working implementation an actual env
pub struct MyEnv;

impl Env for MyEnv {
    type Tensor = TensorData;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor> {
        Ok(TensorData::new(vec![0., 0.], vec![2]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>> {
        let state = TensorData::new(vec![0., 0.], vec![2]);
        let reward = 0.;
        let terminated = false;
        let truncated = false;
        let snapshot = Snapshot::new(state, reward, terminated, truncated);
        Ok(snapshot)
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
        let observation_space = Space::continuous(2, None, None);
        let action_space = Space::Discrete(2);
        EnvDescription::new(observation_space, action_space)
    }
}

struct MyEnvBuilder;

impl EnvBuilder for MyEnvBuilder {
    type Env = MyEnv;

    fn build_env(&self) -> Result<Self::Env> {
        Ok(MyEnv)
    }
}

fn build_env() -> Result<MyEnv> {
    Ok(MyEnv)
}

fn main() {
    // Anything that implement Into<GymEnvBuilder> can be used with the PPO2AlgorithmBuilder::gym
    // method. This includes &str, String and GymEnvBuilder itself (or your own implementation)
    let ppo_builder0 = PPO2AlgorithmBuilder::gym("Pendulum-v1", 10);
    let _ppo0 = ppo_builder0.build().unwrap();

    // Since GymEnvBuilder is an EnvBuilder, it can be used with PPO2AlgorithmBuilder::new
    let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
    let ppo_builder1 = PPO2AlgorithmBuilder::new(gym_env_builder, 10);
    let _ppo1 = ppo_builder1.build().unwrap();

    // This closure that returns an environment can be used as an environment builder
    let env_builder = || Ok(MyEnv);
    let ppo_builder2 = PPO2AlgorithmBuilder::new(env_builder, 10);
    let _ppo2 = ppo_builder2.build().unwrap();

    // This function that returns an environment can also be used as an environment builder
    let ppo_builder3 = PPO2AlgorithmBuilder::new(build_env, 10);
    let _ppo3 = ppo_builder3.build().unwrap();

    // We can implement our own environment builder to be used with PPO2AlgorithmBuilder::new.
    let ppo_builder4 = PPO2AlgorithmBuilder::new(MyEnvBuilder, 10);
    let _ppo4 = ppo_builder4.build().unwrap();
}
// ANCHOR_END: env_builders
