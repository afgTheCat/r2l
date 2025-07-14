// TODO: we will need to integrate the Evaluator as a hook.
use super::{Env, EnvPool, RolloutMode};
use crate::{
    distributions::Distribution,
    env::{EnvironmentDescription, run_rollout, single_step_env, single_step_env_with_buffer},
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};
use candle_core::{Result, Tensor};

pub struct DummyVecEnv<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: Vec<E>,
    pub env_description: EnvironmentDescription,
}

impl<E: Env> EnvPool for DummyVecEnv<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        for (env_idx, buffer) in self.buffers.iter_mut().enumerate() {
            run_rollout(distribution, &mut self.env[env_idx], rollout_mode, buffer)?;
        }
        Ok(self.buffers.clone())
    }

    fn env_description(&self) -> EnvironmentDescription {
        self.env_description.clone()
    }

    fn num_env(&self) -> usize {
        self.env.len()
    }
}

pub struct Evaluator<E: Env> {
    pub env: E,
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
}

impl<E: Env> Evaluator<E> {
    fn evaluate(&mut self, dist: &impl Distribution) -> Result<()> {
        if self.eval_step < self.eval_freq {
            self.eval_step += 1;
            Ok(())
        } else {
            let mut all_rewards = vec![];
            let mut dones = vec![];
            for _ in 0..self.eval_episodes {
                let mut state = self.env.reset(rand::random())?;
                let mut rollout_buffer = RolloutBuffer::default();
                while let (next_state, false) =
                    single_step_env_with_buffer(dist, &state, &mut self.env, &mut rollout_buffer)?
                {
                    state = next_state;
                }
                all_rewards.push(rollout_buffer.rewards.clone());
                dones.push(rollout_buffer.dones.clone());
            }
            let eps = dones
                .iter()
                .map(|x| x.iter().filter(|y| **y).count())
                .sum::<usize>();
            let total_rewards = all_rewards
                .iter()
                .map(|x| x.iter().sum::<f32>())
                .sum::<f32>();
            println!("estimated rew: {}", total_rewards / eps as f32);
            self.eval_step = 0;
            Ok(())
        }
    }
}

// TODO: this whole thing needs to be a wrapper or something
pub struct DummyVecEnvWithEvaluator<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: Vec<E>,
    pub evaluator: Evaluator<E>,
    pub obs_rms: RunningMeanStd,
    pub ret_rms: RunningMeanStd,
    pub clip_obs: f32,
    pub clip_rew: f32,
    pub epsilon: f32,
    pub returns: Tensor,
    pub gamma: f32,
    pub env_description: EnvironmentDescription,
}

impl<E: Env> DummyVecEnvWithEvaluator<E> {
    fn normalize_obs(&self, obs: Tensor) -> Result<Tensor> {
        let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
        let normalized_obs = (obs
            .broadcast_sub(&self.obs_rms.mean)?
            .broadcast_div(&self.obs_rms.var.broadcast_add(&eps)?))?;
        normalized_obs.clamp(-self.clip_obs, self.clip_obs)
    }

    fn normalize_rew(&self, rew: Tensor) -> Result<Tensor> {
        let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
        let normalized_rew = (rew.broadcast_div(&self.ret_rms.var.broadcast_add(&eps)?.sqrt()?))?;
        normalized_rew.clamp(-self.clip_rew, self.clip_rew)
    }

    fn collect_rollouts2<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        let RolloutMode::StepBound { n_steps } = rollout_mode else {
            unreachable!("Step rollout is not supported yet!");
        };
        let mut env_states = self
            .buffers
            .iter_mut()
            .enumerate()
            .map(|(env_idx, rb)| rb.reset(&mut self.env[env_idx], rand::random()).unwrap())
            .collect::<Vec<_>>();
        for _ in 0..n_steps {
            self.evaluator.evaluate(distribution)?;
            let mut obs = vec![];
            let mut actions = vec![];
            let mut rewards = vec![];
            let mut logps = vec![];
            let mut dones = vec![];
            for (env_idx, env) in self.env.iter_mut().enumerate() {
                let (next_state, action, reward, logp, done) =
                    single_step_env(distribution, &env_states[env_idx], env)?;
                obs.push(next_state);
                actions.push(action);
                rewards.push(reward);
                logps.push(logp);
                dones.push(done);
            }

            // normalize states
            let obs = Tensor::stack(&obs, 0)?;
            self.obs_rms.update(&obs)?;
            let obs = self.normalize_obs(obs)?;
            env_states = obs
                .chunk(self.env.len(), 0)?
                .iter()
                .map(|t| t.squeeze(0).unwrap())
                .collect();

            // update returns
            let rewards = Tensor::from_slice(&rewards, rewards.len(), &self.returns.device())?;
            self.returns = self
                .returns
                .broadcast_mul(&Tensor::full(self.gamma, (), &self.returns.device())?)?
                .add(&rewards)?;
            self.ret_rms.update(&self.returns)?;
            let rewards = self.normalize_rew(rewards)?;
            let rewards = rewards.to_vec1()?;

            // update the rbs
            for (rb_idx, rb) in self.buffers.iter_mut().enumerate() {
                rb.push_step(
                    env_states[rb_idx].clone(),
                    actions[rb_idx].clone(),
                    rewards[rb_idx],
                    dones[rb_idx],
                    logps[rb_idx],
                );
            }
        }
        let final_obs = Tensor::stack(&env_states, 0)?;
        self.obs_rms.update(&final_obs)?;
        let final_obs = self.normalize_obs(final_obs)?;
        env_states = final_obs
            .chunk(self.env.len(), 0)?
            .iter()
            .map(|t| t.squeeze(0).unwrap())
            .collect();
        for (rb_idx, rb) in self.buffers.iter_mut().enumerate() {
            rb.set_last_state(env_states[rb_idx].clone());
        }
        Ok(self.buffers.clone())
    }
}

impl<E: Env> EnvPool for DummyVecEnvWithEvaluator<E> {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        self.collect_rollouts2(distribution, rollout_mode)?;
        Ok(self.buffers.clone())
    }

    fn env_description(&self) -> EnvironmentDescription {
        self.env_description.clone()
    }

    fn num_env(&self) -> usize {
        self.env.len()
    }
}
