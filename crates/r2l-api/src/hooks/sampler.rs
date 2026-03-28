// TODO: we should make Evaluator + Noramlizer not depend on candle
use crate::utils::{evaluator::Evaluator, running_mean::RunningMeanStd};
use candle_core::{Device, Result, Tensor};
use r2l_core::sampler::PreprocessorY;
use r2l_core::sampler::buffer::EditableTrajectoryContainer;
use r2l_core::{distributions::Policy, env::Env};

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
            .broadcast_div(&self.obs_rms.var.broadcast_add(&eps)?.sqrt()?))?;
        normalized_obs.clamp(-self.clip_obs, self.clip_obs)
    }

    fn normalize_rew(&self, rew: Tensor) -> Result<Tensor> {
        let eps = Tensor::full(self.epsilon, (), self.ret_rms.var.device())?;
        let normalized_rew = (rew.broadcast_div(&self.ret_rms.var.broadcast_add(&eps)?.sqrt()?))?;
        normalized_rew.clamp(-self.clip_rew, self.clip_rew)
    }

    fn normalize_buffers<B: EditableTrajectoryContainer<Tensor = Tensor>>(
        &mut self,
        states: &mut [B],
        device: &Device,
    ) -> Result<()> {
        let n_envs = states.len();
        let obs: Vec<_> = states.iter_mut().map(|b| b.pop_last_state()).collect();
        let obs = Tensor::stack(&obs, 0)?;
        self.obs_rms.update(&obs)?;
        let obs = self.normalize_obs(obs)?;
        for (state_idx, obs) in obs.chunk(n_envs, 0)?.into_iter().enumerate() {
            let obs = obs.squeeze(0)?;
            states[state_idx].set_last_state(obs);
        }
        let rewards: Vec<_> = states.iter_mut().map(|buf| buf.pop_last_reward()).collect();
        let rewards = Tensor::from_slice(&rewards, rewards.len(), device)?;
        let gamma = Tensor::full(self.gamma, (), device)?;
        self.returns = self.returns.broadcast_mul(&gamma)?.add(&rewards)?;
        self.ret_rms.update(&self.returns)?;
        let rewards = self.normalize_rew(rewards)?;
        for (rew_idx, rew) in (rewards.to_vec1()? as Vec<f32>).iter().enumerate() {
            states[rew_idx].set_last_reward(*rew);
        }
        Ok(())
    }
}

// TODO: this needs to be reconsidered
impl<B: EditableTrajectoryContainer<Tensor = Tensor>> PreprocessorY<Tensor, B> for EnvNormalizer {
    fn preprocess_states(&mut self, _policy: &dyn Policy<Tensor = Tensor>, buffers: &mut [B]) {
        self.normalize_buffers(buffers, &Device::Cpu).unwrap();
    }
}

impl<B: EditableTrajectoryContainer<Tensor = Tensor>, E: Env<Tensor = Tensor>>
    PreprocessorY<Tensor, B> for Evaluator<E>
{
    fn preprocess_states(&mut self, policy: &dyn Policy<Tensor = Tensor>, buffers: &mut [B]) {
        let n_envs = buffers.len();
        self.evaluate(policy, n_envs).unwrap();
    }
}

pub struct EvaluatorNormalizer<E: Env> {
    pub evaluator: Evaluator<E>,
    pub normalizer: EnvNormalizer,
    pub device: Device,
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

impl<B: EditableTrajectoryContainer<Tensor = Tensor>, E: Env<Tensor = Tensor>>
    PreprocessorY<Tensor, B> for EvaluatorNormalizer<E>
{
    fn preprocess_states(&mut self, policy: &dyn Policy<Tensor = Tensor>, buffers: &mut [B]) {
        let n_envs = buffers.len();
        self.evaluator.evaluate(policy, n_envs).unwrap();
        self.normalizer
            .normalize_buffers(buffers, &self.device)
            .unwrap()
    }
}
