use crate::env_pools::RolloutMode;
use crate::env_pools::{EnvHolder, SequentialVecEnvHooks, run_rollout};
use crate::{
    distributions::Distribution,
    env_pools::{Env, single_step_env},
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::{Result, Tensor};

pub struct VecEnvHolder<E: Env> {
    pub envs: Vec<E>,
    pub buffers: Vec<RolloutBuffer>,
}

impl<E: Env> VecEnvHolder<E> {
    fn single_step_env(
        &mut self,
        distr: &impl Distribution,
        current_states: &mut Vec<Tensor>,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<(Tensor, Tensor, f32, f32, bool)>> {
        let mut states = self
            .envs
            .iter_mut()
            .zip(current_states.iter())
            .map(|(env, state)| single_step_env(distr, state, env))
            .collect::<Result<Vec<_>>>()?;
        // TODO: this is kinda jank like this. Maybe we need the previous state here?
        hooks.step_hook(distr, &mut states)?;
        // save the states. This is super not obvious what is going on here
        for (idx, rb) in self.buffers.iter_mut().enumerate() {
            let state = &current_states[idx];
            let (next_state, action, reward, logp, done) = &states[idx];
            rb.push_step(state.clone(), action.detach(), *reward, *done, *logp);
            current_states[idx] = next_state.clone();
        }
        Ok(states)
    }
}

impl<E: Env> EnvHolder for VecEnvHolder<E> {
    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    // TODO: make this work
    fn sequential_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<RolloutBuffer>> {
        let mut current_states = self
            .buffers
            .iter_mut()
            .enumerate()
            .map(|(env_idx, rb)| rb.reset(&mut self.envs[env_idx]))
            .collect::<Result<Vec<_>>>()?;
        let mut steps_taken = 0;
        let num_envs = self.envs.len();
        match rollout_mode {
            RolloutMode::StepBound { n_steps } => {
                while steps_taken < n_steps {
                    self.single_step_env(distr, &mut current_states, hooks)?;
                    steps_taken += num_envs;
                }
                // TODO: what should this be? The current state?
                hooks.post_step_hook(&mut current_states)?;
                for (rb_idx, rb) in self.buffers.iter_mut().enumerate() {
                    rb.set_last_state(current_states[rb_idx].clone());
                }
            }
            RolloutMode::EpisodeBound { n_episodes } => {
                todo!()
            }
        }
        Ok(self.buffers.clone())
    }

    // not really async in this case, only each environment does the thing after one another
    fn async_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        for (env_idx, buffer) in self.buffers.iter_mut().enumerate() {
            let env = &mut self.envs[env_idx];
            let state = buffer.reset(env)?;
            let (states, last_state) = run_rollout(distr, env, rollout_mode, state)?;
            buffer.set_states(states, last_state);
        }
        Ok(self.buffers.clone())
    }
}
