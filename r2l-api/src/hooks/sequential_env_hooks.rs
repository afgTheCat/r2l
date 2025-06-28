use candle_core::{Device, Result, Tensor};
use r2l_core::{
    distributions::Distribution,
    env::{Env, sequential_vec_env::SequentialVecEnvHooks, single_step_env_with_buffer},
    utils::{rollout_buffer::RolloutBuffer, running_mean_std::RunningMeanStd},
};

pub struct Evaluator<E: Env> {
    pub env: E,
    pub eval_episodes: usize,
    pub eval_freq: usize,
    pub eval_step: usize,
}

impl<E: Env> Evaluator<E> {
    pub fn new(env: E, eval_episodes: usize, eval_freq: usize, eval_step: usize) -> Self {
        Self {
            env,
            eval_episodes,
            eval_freq,
            eval_step,
        }
    }

    fn evaluate(&mut self, dist: &dyn Distribution) -> Result<()> {
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
                    single_step_env_with_buffer(dist, &state, &self.env, &mut rollout_buffer)?
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

pub struct EnvNormalizer {
    pub obs_rms: RunningMeanStd,
    pub ret_rms: RunningMeanStd,
    pub returns: Tensor,
    pub epsilon: f32,
    pub gamma: f32,
    pub clip_obs: f32,
    pub clip_rew: f32,
}

impl EnvNormalizer {
    pub fn new(
        obs_rms: RunningMeanStd,
        ret_rms: RunningMeanStd,
        returns: Tensor,
        epsilon: f32,
        gamma: f32,
        clip_obs: f32,
        clip_rew: f32,
    ) -> Self {
        Self {
            obs_rms,
            ret_rms,
            returns,
            epsilon,
            gamma,
            clip_obs,
            clip_rew,
        }
    }

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
}

pub struct EvaluatorNormalizer<E: Env> {
    evaluator: Evaluator<E>,
    normalizer: EnvNormalizer,
    device: Device,
}

impl<E: Env> EvaluatorNormalizer<E> {
    pub fn new(evaluator: Evaluator<E>, normalizer: EnvNormalizer, device: Device) -> Self {
        Self {
            evaluator,
            normalizer,
            device,
        }
    }
}

impl<E: Env> SequentialVecEnvHooks for EvaluatorNormalizer<E> {
    fn step_hook(
        &mut self,
        dist: &dyn Distribution,
        states: &mut Vec<(Tensor, Tensor, f32, f32, bool)>,
    ) -> candle_core::Result<bool> {
        self.evaluator.evaluate(dist)?;
        let states_len = states.len();
        let obs: Vec<_> = states.iter().map(|(obs, ..)| obs.clone()).collect();
        let obs = Tensor::stack(&obs, 0)?;
        self.normalizer.obs_rms.update(&obs)?;
        let obs = self.normalizer.normalize_obs(obs)?;
        for (state_idx, obs) in obs.chunk(states_len, 0)?.into_iter().enumerate() {
            let obs = obs.squeeze(0)?;
            states[state_idx].0 = obs;
        }

        let rewards: Vec<_> = states.iter().map(|(_, _, rew, ..)| *rew).collect();
        let rewards = Tensor::from_slice(&rewards, rewards.len(), &self.device)?;
        let gamma = Tensor::full(self.normalizer.gamma, (), &self.device)?;
        self.normalizer.returns = self
            .normalizer
            .returns
            .broadcast_mul(&gamma)?
            .add(&rewards)?;

        self.normalizer.ret_rms.update(&self.normalizer.returns)?;
        let rewards = self.normalizer.normalize_rew(rewards)?;
        for (rew_idx, rew) in (rewards.to_vec1()? as Vec<f32>).iter().enumerate() {
            states[rew_idx].2 = *rew;
        }
        Ok(false)
    }

    fn post_step_hook(&mut self, last_states: &mut Vec<Tensor>) -> candle_core::Result<bool> {
        let states_len = last_states.len();
        let final_obs = Tensor::stack(&last_states, 0)?;
        self.normalizer.obs_rms.update(&final_obs)?;
        let final_obs = self.normalizer.normalize_obs(final_obs)?;
        for (state_idx, obs) in final_obs.chunk(states_len, 0)?.into_iter().enumerate() {
            let obs = obs.squeeze(0)?;
            last_states[state_idx] = obs;
        }
        Ok(false)
    }
}

pub struct EmptySequentialVecEnv;

impl SequentialVecEnvHooks for EmptySequentialVecEnv {
    fn step_hook(
        &mut self,
        _: &dyn Distribution,
        _: &mut Vec<(Tensor, Tensor, f32, f32, bool)>,
    ) -> candle_core::Result<bool> {
        Ok(false)
    }

    fn post_step_hook(&mut self, _: &mut Vec<Tensor>) -> candle_core::Result<bool> {
        Ok(false)
    }
}
