use candle_core::Device;
use r2l_core::{
    env::{EnvPoolType, dummy_vec_env::DummyVecEnv},
    utils::rollout_buffer::RolloutBuffer,
};
use r2l_gym::GymEnv;

pub enum VecPoolType {
    Dummy,
    Vec,
    Subprocessing,
}

pub struct EnvPoolBuilder {
    pub pool_type: VecPoolType,
    pub n_envs: usize,
    pub gym_env_name: Option<String>,
}

impl Default for EnvPoolBuilder {
    fn default() -> Self {
        Self {
            pool_type: VecPoolType::Dummy,
            n_envs: 16,
            gym_env_name: None,
        }
    }
}

impl EnvPoolBuilder {
    pub fn build(&mut self, device: &Device) -> EnvPoolType<GymEnv> {
        assert!(self.n_envs >= 1, "At least env should be present");
        let Some(gym_env_name) = &self.gym_env_name else {
            todo!()
        };
        match self.pool_type {
            VecPoolType::Dummy => {
                let buffers = vec![RolloutBuffer::default(); self.n_envs];
                let env = (0..self.n_envs)
                    .map(|_| GymEnv::new(&gym_env_name, None, &device).unwrap())
                    .collect::<Vec<_>>();
                let observation_space = env[0].observation_space();
                let action_space = env[0].action_space();
                EnvPoolType::Dummy(DummyVecEnv {
                    buffers,
                    env,
                    observation_space,
                    action_space,
                })
            }
            _ => todo!(),
        }
    }
}
