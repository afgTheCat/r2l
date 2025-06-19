// TODO: we will need to integrate the Evaluator as a hook.
use super::{Env, EnvPool, RolloutMode};
use crate::{
    distributions::Distribution,
    env::{run_rollout, single_step_env},
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Result;

pub struct DummyVecEnv<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: Vec<E>,
}

impl<E: Env> EnvPool for DummyVecEnv<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        for (env_idx, buffer) in self.buffers.iter_mut().enumerate() {
            run_rollout(distribution, &self.env[env_idx], rollout_mode, buffer, None)?;
        }
        Ok(self.buffers.clone())
    }
}

pub struct Evaluator<E: Env> {
    pub env: E,
    pub eval_episodes: usize,
}

impl<E: Env> Evaluator<E> {
    fn evaluate(&self, dist: &impl Distribution) -> Result<Vec<Vec<f32>>> {
        let mut all_rewards = vec![];
        for _ in 0..self.eval_episodes {
            let mut state = self.env.reset(rand::random())?;
            let mut rollout_buffer = RolloutBuffer::default();
            while let (next_state, false) =
                single_step_env(dist, &state, &self.env, &mut rollout_buffer)?
            {
                state = next_state;
            }
            all_rewards.push(rollout_buffer.rewards.clone());
        }
        Ok(all_rewards)
    }
}

pub struct DummyVecEnvWithEvaluator<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: Vec<E>,
    pub evaluator: Evaluator<E>,
    pub eval_freq: usize,
    pub eval_step: usize,
}

impl<E: Env> EnvPool for DummyVecEnvWithEvaluator<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        match rollout_mode {
            RolloutMode::StepBound { n_steps } => {
                let mut env_states = self
                    .buffers
                    .iter_mut()
                    .enumerate()
                    .map(|(env_idx, rb)| rb.reset(&self.env[env_idx], rand::random()).unwrap())
                    .collect::<Vec<_>>();
                for _ in 0..n_steps {
                    for (env_idx, rb) in self.buffers.iter_mut().enumerate() {
                        let (next_state, _done) = single_step_env(
                            distribution,
                            &env_states[env_idx],
                            &self.env[env_idx],
                            rb,
                        )?;
                        env_states[env_idx] = next_state;
                    }
                    self.eval_step += 1;
                    if self.eval_step == self.eval_freq {
                        self.eval_step = 0;
                        self.evaluator.evaluate(distribution)?;
                    }
                }
            }
            RolloutMode::EpisodeBound { .. } => {
                unreachable!("Step rollout is not supported yet!");
            }
        }
        Ok(self.buffers.clone())
    }
}
