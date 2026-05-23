//! Prototype PPO training path that consumes trajectory batches directly.

use anyhow::Result;
use r2l_core::{
    buffers::buffer::TrajectoryBatch,
    models::{LearningModule, Policy, ValueFunction},
    on_policy::{learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses},
    rng::RNG,
    tensor::{R2lTensor, R2lTensorMath},
};
use rand::seq::SliceRandom;

use crate::{
    HookResult,
    on_policy_algorithms::{Advantages, Logps, Returns},
};

/// Hyperparameters controlling PPO training behavior.
pub struct PPOParams {
    /// Clipping range applied to the PPO policy ratio.
    pub clip_range: f32,
    /// Discount factor used for return and advantage estimation.
    pub gamma: f32,
    /// GAE lambda used for advantage estimation.
    pub lambda: f32,
    /// Minibatch size used during each PPO epoch.
    pub sample_size: usize,
}

impl Default for PPOParams {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            lambda: 0.8,
            gamma: 0.98,
            sample_size: 64,
        }
    }
}

/// Per-minibatch data exposed to [`PPOHook::batch_hook`].
pub struct PPOBatchData<T: R2lTensor> {
    /// Sampled observations in the minibatch.
    pub observations: Vec<T>,
    /// Sampled actions in the minibatch.
    pub actions: Vec<T>,
    /// Current policy log-probabilities for the sampled actions.
    pub logp: T,
    /// Value-function predictions for the sampled observations.
    pub values_pred: T,
    /// Difference between current and old log-probabilities.
    pub logp_diff: T,
    /// Probability ratio `exp(logp_diff)` used by the PPO objective.
    pub ratio: T,
}

/// Hook interface for customizing PPO training over [`TrajectoryBatch`] inputs.
pub trait PPOHook<M: OnPolicyLearningModule> {
    fn before_learning_hook(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _batches: &[TrajectoryBatch<'_, M::InferenceTensor>],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _batches: &[TrajectoryBatch<'_, M::InferenceTensor>],
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _params: &mut PPOParams,
        _module: &mut M,
        _losses: &mut <M as LearningModule>::Losses,
        _data: &PPOBatchData<M::LearningTensor>,
    ) -> anyhow::Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

/// Prototype PPO variant over finalized trajectory batches.
pub struct PPO2<Module: OnPolicyLearningModule, Hooks: PPOHook<Module>> {
    /// PPO hyperparameters.
    pub params: PPOParams,
    /// Learning module containing policy, value function, and optimizer state.
    pub lm: Module,
    /// Hook implementation used to customize learning behavior.
    pub hooks: Hooks,
}

impl<Module: OnPolicyLearningModule, Hooks: PPOHook<Module>> PPO2<Module, Hooks> {
    fn batch_loop(
        &mut self,
        batches: &[TrajectoryBatch<'_, Module::InferenceTensor>],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> anyhow::Result<()> {
        let mut index_iterator = BatchIndexIterator::new(batches, self.params.sample_size);
        let lm = &mut self.lm;
        loop {
            let Some(indices) = index_iterator.iter() else {
                return Ok(());
            };
            let (observations, actions) = sample(batches, &indices, Module::lifter);
            let advantages = lm.tensor_from_slice(&advantages.sample(&indices));
            let logp_old = lm.tensor_from_slice(&logps.sample(&indices));
            let returns = lm.tensor_from_slice(&returns.sample(&indices));
            let logp = lm.policy().log_probs(&observations, &actions)?;
            let values_pred = lm.values(&observations)?;
            let value_loss = returns.sub(&values_pred)?.sqr()?.mean()?;
            let logp_diff = logp.sub(&logp_old)?;
            let ratio = logp_diff.exp()?;
            let clip_ratio =
                ratio.clamp(1. - self.params.clip_range, 1. + self.params.clip_range)?;
            let clipped_adv = clip_ratio.mul(&advantages)?;
            let ratio_adv = ratio.mul(&advantages)?;
            let policy_loss = ratio_adv.minimum(&clipped_adv)?.neg()?.mean()?;
            let mut losses = Module::Losses::from_policy_value_losses(policy_loss, value_loss);
            let ppo_data = PPOBatchData {
                observations,
                actions,
                logp,
                values_pred,
                logp_diff,
                ratio,
            };
            r2l_core::return_on_hook_result!(self.hooks.batch_hook(
                &mut self.params,
                lm,
                &mut losses,
                &ppo_data
            )?);
            lm.update(losses)?;
        }
    }

    fn learning_loop(
        &mut self,
        batches: &[TrajectoryBatch<'_, Module::InferenceTensor>],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> anyhow::Result<()> {
        loop {
            self.batch_loop(batches, &advantages, &logps, &returns)?;
            let rollout_hook_res = self
                .hooks
                .rollout_hook(&mut self.params, &mut self.lm, batches);
            r2l_core::return_on_hook_result!(rollout_hook_res?);
        }
    }

    /// Prototype learning entrypoint over finalized trajectory batches.
    pub fn learn(
        &mut self,
        batches: &[TrajectoryBatch<'_, Module::InferenceTensor>],
    ) -> Result<()> {
        let (mut advantages, mut returns) = batches_advantages_and_returns(
            batches,
            &self.lm,
            self.params.gamma,
            self.params.lambda,
            Module::lifter,
        )?;
        r2l_core::return_on_hook_result!(self.hooks.before_learning_hook(
            &mut self.params,
            &mut self.lm,
            batches,
            &mut advantages,
            &mut returns
        )?);
        let actor = self.lm.inference_policy();
        let logps = logps(batches, &actor)?;
        self.learning_loop(batches, advantages, returns, logps)?;
        Ok(())
    }
}

fn sample<T1: R2lTensor, T2: R2lTensor, L: Fn(&T1) -> T2>(
    batches: &[TrajectoryBatch<'_, T1>],
    indices: &[(usize, usize)],
    lifter: L,
) -> (Vec<T2>, Vec<T2>) {
    let mut observations = vec![];
    let mut actions = vec![];
    for (batch_idx, idx) in indices {
        observations.push(lifter(&batches[*batch_idx].states()[*idx]));
        actions.push(lifter(&batches[*batch_idx].actions()[*idx]));
    }
    (observations, actions)
}

fn logps<T: R2lTensor>(
    batches: &[TrajectoryBatch<'_, T>],
    policy: &impl Policy<Tensor = T>,
) -> anyhow::Result<Logps> {
    let mut logps = vec![];
    for batch in batches {
        let logp = policy
            .log_probs(batch.states(), batch.actions())
            .map(|t| t.to_vec())?;
        logps.push(logp);
    }
    Ok(Logps(logps))
}

fn batch_advantages_and_returns<T1: R2lTensor, T2: R2lTensor, L: Fn(&T1) -> T2>(
    batch: &TrajectoryBatch<'_, T1>,
    value_func: &impl ValueFunction<Tensor = T2>,
    gamma: f32,
    lambda: f32,
    lifter: L,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let mut states = batch.states().iter().map(&lifter).collect::<Vec<_>>();
    states.push(lifter(batch.next_states().last().unwrap()));
    let values_stacked = value_func.values(&states)?;
    let values: Vec<f32> = values_stacked.to_vec();
    let total_steps = batch.rewards().len();
    let mut advantages: Vec<f32> = vec![0.; total_steps];
    let mut returns: Vec<f32> = vec![0.; total_steps];
    let mut last_gae_lam: f32 = 0.;

    for i in (0..total_steps).rev() {
        let done = batch.terminated()[i] || batch.truncated()[i];
        let next_non_terminal = if done {
            last_gae_lam = 0.;
            0f32
        } else {
            1.
        };
        let delta = batch.rewards()[i] + next_non_terminal * gamma * values[i + 1] - values[i];
        last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
        advantages[i] = last_gae_lam;
        returns[i] = last_gae_lam + values[i];
    }
    Ok((advantages, returns))
}

fn batches_advantages_and_returns<T1: R2lTensor, T2: R2lTensor, L: Fn(&T1) -> T2>(
    batches: &[TrajectoryBatch<'_, T1>],
    value_func: &impl ValueFunction<Tensor = T2>,
    gamma: f32,
    lambda: f32,
    lifter: L,
) -> anyhow::Result<(Advantages, Returns)> {
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];
    for batch in batches {
        let (advantages, returns) =
            batch_advantages_and_returns(batch, value_func, gamma, lambda, &lifter)?;
        advantage_vec.push(advantages);
        returns_vec.push(returns);
    }
    Ok((Advantages(advantage_vec), Returns(returns_vec)))
}

struct BatchIndexIterator {
    indices: Vec<(usize, usize)>,
    sample_size: usize,
    current: usize,
}

impl BatchIndexIterator {
    fn new<T: R2lTensor>(batches: &[TrajectoryBatch<'_, T>], sample_size: usize) -> Self {
        let mut indices = (0..batches.len())
            .flat_map(|i| {
                let batch = &batches[i];
                (0..batch.len()).map(|j| (i, j)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        RNG.with_borrow_mut(|rng| indices.shuffle(rng));
        Self {
            indices,
            sample_size,
            current: 0,
        }
    }

    fn iter(&mut self) -> Option<Vec<(usize, usize)>> {
        let total_size = self.indices.len();
        if self.sample_size + self.current > total_size {
            return None;
        }
        let batch_indices = &self.indices[self.current..self.current + self.sample_size];
        self.current += self.sample_size;
        Some(batch_indices.to_owned())
    }
}
