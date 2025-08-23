use candle_core::{Device, Result, Tensor};
use r2l_core::distributions::Distribution;
use r2l_core::env::{Env, SnapShot};
use r2l_core::numeric::Buffer;
use r2l_gym::GymEnv;

pub fn run_gym_episodes(
    env: &str,
    ep_count: usize,
    dist: &impl Distribution<Observation = Tensor, Action = Tensor, Entropy = Tensor>,
) -> Result<()> {
    for _ in 0..ep_count {
        let device = Device::Cpu;
        let mut env = GymEnv::new(env, Some("human".into()));
        let seed = rand::random();
        let mut state = env.reset(seed).to_candle_tensor(&device);
        let mut action = dist.get_action(state.unsqueeze(0)?)?;
        while let SnapShot {
            state: next_state,
            terminated: false,
            trancuated: false,
            ..
        } = env.step(&Buffer::from_candle_tensor(&action))
        {
            state = next_state.to_candle_tensor(&device);
            // TODO: unsqueeze seems too much here
            let next_action = dist.get_action(state.unsqueeze(0)?)?;
            action = next_action;
        }
    }
    Ok(())
}
