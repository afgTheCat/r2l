// builders + hooks + higher level helpers
pub mod builders;
pub mod hooks;

use candle_core::{Device, Result};
use r2l_core::distributions::Distribution;
use r2l_core::env::Env;
use r2l_gym::GymEnv;

pub fn run_gym_episodes(env: &str, ep_count: usize, dist: &impl Distribution) -> Result<()> {
    for _ in 0..ep_count {
        let device = Device::Cpu;
        let env = GymEnv::new(env, Some("human".into()), &device)?;
        let mut state = env.reset(rand::random())?;
        let (mut action, _) = dist.get_action(&state.unsqueeze(0)?)?;
        while let (next_state, _, false, false) = env.step(&action)? {
            state = next_state;
            // TODO: unsqueeze seems too much here
            let (next_action, _) = dist.get_action(&state.unsqueeze(0)?)?;
            action = next_action;
        }
    }
    Ok(())
}
