// TODO: we will need to integrate the Evaluator as a hook.
use super::{Env, EnvPool, RolloutMode};
use crate::{
    distributions::Distribution,
    env::{Space, run_rollout, single_step_env, single_step_env_with_buffer},
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};
use candle_core::{Result, Tensor};

pub struct DummyVecEnv<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: Vec<E>,
    pub action_space: Space,
    pub observation_space: Space,
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

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn observation_space(&self) -> Space {
        self.observation_space.clone()
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
                single_step_env_with_buffer(dist, &state, &self.env, &mut rollout_buffer)?
            {
                state = next_state;
            }
            all_rewards.push(rollout_buffer.rewards.clone());
        }
        Ok(all_rewards)
    }
}

// TODO: this whole thing needs to be a wrapper or something
pub struct DummyVecEnvWithEvaluator<E: Env> {
    pub buffers: Vec<RolloutBuffer>,
    pub env: Vec<E>,
    pub evaluator: Evaluator<E>,
    pub eval_freq: usize,
    pub eval_step: usize,
    pub obs_rms: RunningMeanStd,
    pub ret_rms: RunningMeanStd,
    pub clip_obs: f32,
    pub clip_rew: f32,
    pub epsilon: f32,
    pub returns: Tensor,
    pub gamma: f32,
    pub action_space: Space,
    pub observation_space: Space,
}

impl<E: Env> DummyVecEnvWithEvaluator<E> {
    fn normalize_obs(&self, obs: Tensor) -> Result<Tensor> {
        let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
        let normalized_obs =
            (obs.sub(&self.obs_rms.mean) / &self.obs_rms.var.broadcast_add(&eps)?)?;
        normalized_obs.clamp(-self.clip_obs, self.clip_obs)
    }

    fn normalize_rew(&self, rew: Tensor) -> Result<Tensor> {
        let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
        let normalized_rew = (rew / self.ret_rms.var.broadcast_add(&eps)?.sqrt()?)?;
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
            .map(|(env_idx, rb)| rb.reset(&self.env[env_idx], rand::random()).unwrap())
            .collect::<Vec<_>>();
        for _ in 0..n_steps {
            let mut obs = vec![];
            let mut actions = vec![];
            let mut rewards = vec![];
            let mut logps = vec![];
            let mut dones = vec![];
            for (env_idx, env) in self.env.iter().enumerate() {
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
            env_states = obs.chunk(self.env.len(), 0)?;

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
        // match rollout_mode {
        //     RolloutMode::StepBound { n_steps } => {
        //         let mut env_states = self
        //             .buffers
        //             .iter_mut()
        //             .enumerate()
        //             .map(|(env_idx, rb)| rb.reset(&self.env[env_idx], rand::random()).unwrap())
        //             .collect::<Vec<_>>();
        //         for _ in 0..n_steps {
        //             let mut states = vec![];
        //             for (env_idx, rb) in self.buffers.iter_mut().enumerate() {
        //                 let (next_state, _done) = single_step_env_with_buffer(
        //                     distribution,
        //                     &env_states[env_idx],
        //                     &self.env[env_idx],
        //                     rb,
        //                 )?;
        //                 states.push(next_state);
        //             }
        //             let obs = Tensor::stack(&states, 0)?;
        //             self.obs_rms.update(&obs)?;
        //             let obs = self.normalize_obs(obs)?;
        //             // TODO: this obs should update the one in the Rollout Buffer
        //             for (env_idx, state) in states.into_iter().enumerate() {
        //                 env_states[env_idx] = state;
        //             }
        //             self.eval_step += 1;
        //             if self.eval_step == self.eval_freq {
        //                 self.eval_step = 0;
        //                 self.evaluator.evaluate(distribution)?;
        //             }
        //         }
        //     }
        //     RolloutMode::EpisodeBound { .. } => {
        //         unreachable!("Step rollout is not supported yet!");
        //     }
        // }
        Ok(self.buffers.clone())
    }

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn observation_space(&self) -> Space {
        self.observation_space.clone()
    }
}
