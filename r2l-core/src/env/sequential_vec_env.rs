use super::Env;
use crate::{
    distributions::Distribution,
    env::{EnvPool, EnvironmentDescription, RolloutMode, single_step_env},
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Result, Tensor};

pub trait SequentialVecEnvHooks {
    fn step_hook(
        &mut self,
        distribution: &dyn Distribution,
        states: &mut Vec<(Tensor, Tensor, f32, f32, bool)>,
    ) -> candle_core::Result<bool>;

    fn post_step_hook(&mut self, last_states: &mut Vec<Tensor>) -> candle_core::Result<bool>;
}

// TODO: will be renamed as sequential VecEnv
pub struct SequentialVecEnv<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub envs: Vec<E>,
    pub env_description: EnvironmentDescription,
    pub hooks: Box<dyn SequentialVecEnvHooks>,
}

impl<E: Env> SequentialVecEnv<E> {
    fn collect_step_bound_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        env_states: &mut Vec<Tensor>,
        n_steps: usize,
    ) -> Result<Vec<RolloutBuffer>> {
        for _ in 0..n_steps {
            let mut states = self
                .envs
                .iter()
                .zip(env_states.iter())
                .map(|(env, state)| single_step_env(distribution, state, env))
                .collect::<Result<Vec<_>>>()?;
            self.hooks.step_hook(distribution, &mut states)?;
            for (idx, rb) in self.buffers.iter_mut().enumerate() {
                let state = &env_states[idx];
                let (next_state, action, reward, logp, done) = &states[idx];
                rb.push_step(state.clone(), action.clone(), *reward, *done, *logp);
                env_states[idx] = next_state.clone();
            }
        }
        // TODO: having a separate trait for this is suboptimal
        self.hooks.post_step_hook(env_states)?;
        for (rb_idx, rb) in self.buffers.iter_mut().enumerate() {
            rb.set_last_state(env_states[rb_idx].clone());
        }
        Ok(self.buffers.clone())
    }
}

impl<E: Env> EnvPool for SequentialVecEnv<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        let mut env_states = self
            .buffers
            .iter_mut()
            .enumerate()
            .map(|(env_idx, rb)| rb.reset(&self.envs[env_idx], rand::random()))
            .collect::<Result<Vec<_>>>()?;
        match rollout_mode {
            RolloutMode::StepBound { n_steps } => {
                self.collect_step_bound_rollouts(distribution, &mut env_states, n_steps)
            }
            RolloutMode::EpisodeBound {
                n_episodes: n_steps,
            } => {
                todo!()
            }
        }
    }

    fn env_description(&self) -> EnvironmentDescription {
        self.env_description.clone()
    }
}
