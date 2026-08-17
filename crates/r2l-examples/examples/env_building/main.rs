// ANCHOR: env_builders
use r2l::{Env, EnvBuilder, EnvDescription, PPOAlgorithmBuilder, Snapshot, Space, VecTensor};
use r2l_gym::GymEnvBuilder;

// Not a working implementation an actual env
pub struct MyEnv;

impl Env for MyEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor, r2l_core::error::Error> {
        Ok(VecTensor::new(vec![0., 0.], vec![2])?)
    }

    fn step(
        &mut self,
        _action: Self::Tensor,
    ) -> Result<Snapshot<Self::Tensor>, r2l_core::error::Error> {
        let state = VecTensor::new(vec![0., 0.], vec![2])?;
        let reward = 0.;
        let terminated = false;
        let truncated = false;
        let snapshot = Snapshot::new(state, reward, terminated, truncated);
        Ok(snapshot)
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
        let observation_space = Space::Box {
            min: None,
            max: None,
            shape: vec![2],
        };
        let action_space = Space::Discrete(2);
        EnvDescription::new(observation_space, action_space)
    }
}

struct MyEnvBuilder;

impl EnvBuilder for MyEnvBuilder {
    type Env = MyEnv;

    fn build_env(&self) -> Result<Self::Env, r2l_core::error::Error> {
        Ok(MyEnv)
    }
}

#[allow(clippy::unnecessary_wraps)]
fn build_env() -> Result<MyEnv, r2l_core::error::Error> {
    Ok(MyEnv)
}

fn main() {
    // Anything that implement Into<GymEnvBuilder> can be used with the PPOAlgorithmBuilder::gym
    // method. This includes &str, String and GymEnvBuilder itself (or your own implementation)
    let ppo_builder0 = PPOAlgorithmBuilder::gym("Pendulum-v1", 10).unwrap();
    let _ppo0 = ppo_builder0.build().unwrap();

    // Since GymEnvBuilder is an EnvBuilder, it can be used with PPOAlgorithmBuilder::new
    let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
    let ppo_builder1 = PPOAlgorithmBuilder::new(gym_env_builder, 10).unwrap();
    let _ppo1 = ppo_builder1.build().unwrap();

    // This closure that returns an environment can be used as an environment builder
    let env_builder = || Ok(MyEnv);
    let ppo_builder = PPOAlgorithmBuilder::new(env_builder, 10).unwrap();
    let _ppo = ppo_builder.build().unwrap();

    // This function that returns an environment can also be used as an environment builder
    let ppo_builder3 = PPOAlgorithmBuilder::new(build_env, 10).unwrap();
    let _ppo3 = ppo_builder3.build().unwrap();

    // We can implement our own environment builder to be used with PPOAlgorithmBuilder::new.
    let ppo_builder4 = PPOAlgorithmBuilder::new(MyEnvBuilder, 10).unwrap();
    let _ppo4 = ppo_builder4.build().unwrap();
}
// ANCHOR_END: env_builders
