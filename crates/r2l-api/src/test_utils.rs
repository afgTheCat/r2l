use anyhow::Result;
use candle_core::{Device, Tensor};
use r2l_core::distributions::Policy;
use r2l_core::env::{Env, SnapShot};
use r2l_core::tensor::R2lBuffer;
use r2l_gym::GymEnv;

pub fn run_gym_episodes(
    env: &str,
    ep_count: usize,
    dist: &impl Policy<Tensor = Tensor>,
) -> Result<()> {
    for _ in 0..ep_count {
        let device = Device::Cpu;
        let mut env = GymEnv::new(env, Some("human".into()));
        let seed = rand::random();
        let mut state: Tensor = env.reset(seed)?.into();
        let mut action = dist.get_action(state.unsqueeze(0)?)?;
        while let SnapShot {
            state: next_state,
            terminated: false,
            trancuated: false,
            ..
        } = env.step(R2lBuffer::from_candle_tensor(&action))?
        {
            state = next_state.into();
            // TODO: unsqueeze seems too much here
            let next_action = dist.get_action(state.unsqueeze(0)?)?;
            action = next_action;
        }
    }
    Ok(())
}
