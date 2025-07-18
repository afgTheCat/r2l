use candle_core::{Device, Result, Tensor};
use r2l_api::builders::env_pool::{NormalizerOptions, SequentialEnvHookTypes, VecPoolType};
use r2l_core::env::{Env, EnvironmentDescription, Space};

struct DummyRewardEnv {
    t: u64,
    returned_rewards: [f32; 4],
    returned_reward_idx: usize,
}

impl DummyRewardEnv {
    fn new(returned_reward_idx: usize) -> Result<Self> {
        Ok(Self {
            t: returned_reward_idx as u64,
            returned_rewards: [0., 1., 2., 3.],
            returned_reward_idx,
        })
    }
}

impl Env for DummyRewardEnv {
    fn step(&mut self, action: &Tensor) -> Result<(Tensor, f32, bool, bool)> {
        self.t += 1;
        let index = (self.t as usize + self.returned_reward_idx) % 4;
        let returned_value = self.returned_rewards[index];
        let truncated = self.t == 4;
        let state = Tensor::full(returned_value, (), &Device::Cpu)?;
        Ok((state, returned_value, false, truncated))
    }

    fn reset(&mut self, seed: u64) -> Result<Tensor> {
        self.t = 0;
        let state = Tensor::full(
            self.returned_rewards[self.returned_reward_idx],
            (),
            &Device::Cpu,
        )?;
        Ok(state)
    }

    fn env_description(&self) -> EnvironmentDescription {
        let min = Tensor::full(-1., (), &Device::Cpu).unwrap();
        let max = Tensor::full(1., (), &Device::Cpu).unwrap();
        EnvironmentDescription::new(
            Space::Discrete(2),
            Space::Continous {
                min: Some(min),
                max: Some(max),
                size: 1,
            },
        )
    }
}

#[test]
fn test_obs_rms_vec_normalize() -> Result<()> {
    let env_builder = vec![|| DummyRewardEnv::new(0), || DummyRewardEnv::new(1)];
    let env_pool_builder = VecPoolType::Sequential(SequentialEnvHookTypes::NormalizerOnly {
        options: NormalizerOptions::default(),
    });
    let env_pool = env_pool_builder.build_with_builders(&Device::Cpu, env_builder)?;
    Ok(())
}
