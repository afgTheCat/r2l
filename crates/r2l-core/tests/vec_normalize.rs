use anyhow::Result;
use candle_core::{Device, Tensor};
use r2l_core::{
    env::{Env, EnvironmentDescription, SnapShot, Space},
    tensor::R2lBuffer,
};

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
    type Tensor = R2lBuffer;

    fn step(&mut self, action: R2lBuffer) -> Result<SnapShot<R2lBuffer>> {
        self.t += 1;
        let index = (self.t as usize + self.returned_reward_idx) % 4;
        let returned_value = self.returned_rewards[index];
        let truncated = self.t == 4;
        let state = Tensor::full(returned_value, (), &Device::Cpu).unwrap();
        let snapshot = SnapShot {
            state: R2lBuffer::from_candle_tensor(&state),
            reward: returned_value,
            terminated: false,
            trancuated: truncated,
        };
        Ok(snapshot)
    }

    fn reset(&mut self, seed: u64) -> Result<R2lBuffer> {
        self.t = 0;
        let state = Tensor::full(
            self.returned_rewards[self.returned_reward_idx],
            (),
            &Device::Cpu,
        )
        .unwrap();
        let state = R2lBuffer::from_candle_tensor(&state);
        Ok(state)
    }

    fn env_description(&self) -> EnvironmentDescription<R2lBuffer> {
        let min = Tensor::full(-1., (), &Device::Cpu).unwrap();
        let max = Tensor::full(1., (), &Device::Cpu).unwrap();
        EnvironmentDescription::new(
            Space::Discrete(2),
            Space::Continous {
                min: Some(R2lBuffer::from_candle_tensor(&min)),
                max: Some(R2lBuffer::from_candle_tensor(&max)),
                size: 1,
            },
        )
    }
}

#[test]
fn test_obs_rms_vec_normalize() -> Result<()> {
    let env_builders = vec![|_: &Device| DummyRewardEnv::new(0), |_: &Device| {
        DummyRewardEnv::new(1)
    }];
    // let env_pool_builder = VecPoolType::Sequential(SequentialEnvHookTypes::NormalizerOnly {
    //     options: NormalizerOptions::default(),
    // });
    // let env_pool = env_pool_builder.build_with_builders(&Device::Cpu, env_builders)?;
    Ok(())
}
