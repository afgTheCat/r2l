//! Prototype recurrent PPO training over contiguous trajectory sequences.

use anyhow::{Result, ensure};
use r2l_core::{
    buffers::TrajectoryBatch,
    models::RecurrentPolicy,
    on_policy::{
        algorithm::Agent, learning_module::OnPolicyLearningModule, losses::FromPolicyValueLosses,
    },
    rng::with_rng,
    tensor::R2lTensor,
};
use rand::seq::SliceRandom;

use crate::{
    HookResult,
    on_policy_algorithms::{Advantages, Logps, Returns, batches_advantages_and_returns},
};

/// Hyperparameters controlling recurrent PPO training.
pub struct RecurrentPPOParams {
    /// Clipping range applied to the PPO policy ratio.
    pub clip_range: f32,
    /// Discount factor used for return and advantage estimation.
    pub gamma: f32,
    /// GAE lambda used for advantage estimation.
    pub lambda: f32,
    /// Maximum number of contiguous transitions in one truncated-BPTT update.
    pub sequence_length: usize,
}

impl Default for RecurrentPPOParams {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            lambda: 0.8,
            gamma: 0.98,
            sequence_length: 64,
        }
    }
}

/// Per-sequence data exposed to [`RecurrentPPOHook::batch_hook`].
pub struct RecurrentPPOBatchData<T: R2lTensor> {
    /// Contiguous observations in temporal order.
    pub observations: Vec<T>,
    /// Actions aligned with `observations`.
    pub actions: Vec<T>,
    /// Current policy log-probabilities for the actions.
    pub logp: T,
    /// Per-step action-distribution entropy from the recurrent forward pass.
    pub entropy: T,
    /// Value-function predictions for the observations.
    pub values_pred: T,
    /// Difference between current and old log-probabilities.
    pub logp_diff: T,
    /// Probability ratio `exp(logp_diff)` used by the PPO objective.
    pub ratio: T,
}

/// Hook interface for customizing recurrent PPO training.
pub trait RecurrentPPOHook<M: OnPolicyLearningModule>
where
    M::Policy: RecurrentPolicy<State = M::LearningState>,
    M::InferencePolicy: RecurrentPolicy<State = M::InferenceState>,
{
    fn before_learning_hook<B: TrajectoryBatch<M::InferenceTensor, State = M::InferenceState>>(
        &mut self,
        _params: &mut RecurrentPPOParams,
        _module: &mut M,
        _batches: &[B],
        _advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> Result<HookResult> {
        Ok(HookResult::Continue)
    }

    fn rollout_hook<B: TrajectoryBatch<M::InferenceTensor, State = M::InferenceState>>(
        &mut self,
        _params: &mut RecurrentPPOParams,
        _module: &mut M,
        _batches: &[B],
    ) -> Result<HookResult> {
        Ok(HookResult::Break)
    }

    fn batch_hook(
        &mut self,
        _params: &mut RecurrentPPOParams,
        _module: &mut M,
        _losses: &mut M::Losses,
        _data: &RecurrentPPOBatchData<M::LearningTensor>,
    ) -> Result<HookResult> {
        Ok(HookResult::Continue)
    }
}

impl<M> RecurrentPPOHook<M> for ()
where
    M: OnPolicyLearningModule,
    M::Policy: RecurrentPolicy<State = M::LearningState>,
    M::InferencePolicy: RecurrentPolicy<State = M::InferenceState>,
{
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SequenceIndex {
    batch: usize,
    start: usize,
    end: usize,
}

impl SequenceIndex {
    fn indices(self) -> impl Iterator<Item = (usize, usize)> {
        (self.start..self.end).map(move |step| (self.batch, step))
    }
}

struct SequenceIndexIterator {
    sequences: Vec<SequenceIndex>,
}

impl SequenceIndexIterator {
    fn sequences<T: R2lTensor, B: TrajectoryBatch<T>>(
        batches: &[B],
        sequence_length: usize,
    ) -> Vec<SequenceIndex> {
        let mut sequences = Vec::new();
        for (batch_idx, batch) in batches.iter().enumerate() {
            let mut start = 0;
            while start < batch.len() {
                let remaining_episode = batch.terminated()[start..]
                    .iter()
                    .zip(&batch.truncated()[start..])
                    .position(|(terminated, truncated)| *terminated || *truncated)
                    .map_or(batch.len() - start, |offset| offset + 1);
                let episode_end = start + remaining_episode;
                while start < episode_end {
                    let end = (start + sequence_length).min(episode_end);
                    sequences.push(SequenceIndex {
                        batch: batch_idx,
                        start,
                        end,
                    });
                    start = end;
                }
            }
        }
        sequences
    }

    fn new<T: R2lTensor, B: TrajectoryBatch<T>>(batches: &[B], sequence_length: usize) -> Self {
        let mut sequences = Self::sequences(batches, sequence_length);
        with_rng(|rng| sequences.shuffle(rng));
        Self { sequences }
    }
}

impl Iterator for SequenceIndexIterator {
    type Item = SequenceIndex;

    fn next(&mut self) -> Option<Self::Item> {
        self.sequences.pop()
    }
}

fn sample_sequence<T1, T2, B, L>(
    batches: &[B],
    sequence: SequenceIndex,
    lifter: L,
) -> (Vec<T2>, Vec<T2>)
where
    T1: R2lTensor,
    T2: R2lTensor,
    B: TrajectoryBatch<T1>,
    L: Fn(&T1) -> T2,
{
    let batch = &batches[sequence.batch];
    let observations = batch.states()[sequence.start..sequence.end]
        .iter()
        .map(&lifter)
        .collect();
    let actions = batch.actions()[sequence.start..sequence.end]
        .iter()
        .map(lifter)
        .collect();
    (observations, actions)
}

fn recurrent_logps<T, B, P>(batches: &[B], policy: &P, sequence_length: usize) -> Result<Logps>
where
    T: R2lTensor,
    B: TrajectoryBatch<T, State = P::State>,
    P: RecurrentPolicy<Tensor = T>,
{
    let mut logps = batches
        .iter()
        .map(|batch| vec![0.0; batch.len()])
        .collect::<Vec<_>>();
    for sequence in SequenceIndexIterator::sequences(batches, sequence_length) {
        let batch = &batches[sequence.batch];
        let output = policy.evaluate_sequence(
            &batch.states()[sequence.start..sequence.end],
            &batch.actions()[sequence.start..sequence.end],
            batch.actor_states()[sequence.start].as_ref(),
        )?;
        let sequence_logps = output.log_probs.to_vec();
        ensure!(
            sequence_logps.len() == sequence.end - sequence.start,
            "recurrent policy returned one log probability per sequence step"
        );
        logps[sequence.batch][sequence.start..sequence.end].copy_from_slice(&sequence_logps);
    }
    Ok(Logps(logps))
}

/// PPO variant that trains recurrent policies over contiguous sequence chunks.
pub struct RecurrentPPO<Module, Hooks>
where
    Module: OnPolicyLearningModule,
    Module::Policy: RecurrentPolicy<State = Module::LearningState>,
    Module::InferencePolicy: RecurrentPolicy<State = Module::InferenceState>,
    Hooks: RecurrentPPOHook<Module>,
{
    /// Recurrent PPO hyperparameters.
    pub params: RecurrentPPOParams,
    /// Learning module containing policy, value function, and optimizer state.
    pub lm: Module,
    /// Hook implementation used to customize learning behavior.
    pub hooks: Hooks,
}

impl<Module, Hooks> RecurrentPPO<Module, Hooks>
where
    Module: OnPolicyLearningModule,
    Module::Policy: RecurrentPolicy<State = Module::LearningState>,
    Module::InferencePolicy: RecurrentPolicy<State = Module::InferenceState>,
    Hooks: RecurrentPPOHook<Module>,
{
    fn batch_loop<B: TrajectoryBatch<Module::InferenceTensor, State = Module::InferenceState>>(
        &mut self,
        batches: &[B],
        advantages: &Advantages,
        logps: &Logps,
        returns: &Returns,
    ) -> Result<()> {
        let sequences = SequenceIndexIterator::new(batches, self.params.sequence_length);
        let lm = &mut self.lm;
        for sequence in sequences {
            let indices = sequence.indices().collect::<Vec<_>>();
            let (observations, actions) = sample_sequence(batches, sequence, Module::lifter);
            let advantages = lm.tensor_from_slice(&advantages.sample(&indices));
            let logp_old = lm.tensor_from_slice(&logps.sample(&indices));
            let returns = lm.tensor_from_slice(&returns.sample(&indices));
            let initial_state = batches[sequence.batch].actor_states()[sequence.start]
                .as_ref()
                .map(Module::state_lifter);
            let evaluation =
                lm.policy()
                    .evaluate_sequence(&observations, &actions, initial_state.as_ref())?;
            let logp = evaluation.log_probs;
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
            let data = RecurrentPPOBatchData {
                observations,
                actions,
                logp,
                entropy: evaluation.entropy,
                values_pred,
                logp_diff,
                ratio,
            };
            r2l_core::return_on_hook_result!(self.hooks.batch_hook(
                &mut self.params,
                lm,
                &mut losses,
                &data
            )?);
            lm.update(losses)?;
        }
        Ok(())
    }

    fn learning_loop<
        B: TrajectoryBatch<Module::InferenceTensor, State = Module::InferenceState>,
    >(
        &mut self,
        batches: &[B],
        advantages: Advantages,
        returns: Returns,
        logps: Logps,
    ) -> Result<()> {
        loop {
            self.batch_loop(batches, &advantages, &logps, &returns)?;
            let result = self
                .hooks
                .rollout_hook(&mut self.params, &mut self.lm, batches)?;
            r2l_core::return_on_hook_result!(result);
        }
    }

    /// Learns from finalized trajectory batches using truncated BPTT.
    pub fn learn<B: TrajectoryBatch<Module::InferenceTensor, State = Module::InferenceState>>(
        &mut self,
        batches: &[B],
    ) -> Result<()> {
        ensure!(
            self.params.sequence_length > 0,
            "sequence length must be positive"
        );
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
        let logps = recurrent_logps(batches, &actor, self.params.sequence_length)?;
        self.learning_loop(batches, advantages, returns, logps)
    }
}

impl<M, H> Agent for RecurrentPPO<M, H>
where
    M: OnPolicyLearningModule,
    M::Policy: RecurrentPolicy<State = M::LearningState>,
    M::InferencePolicy: RecurrentPolicy<State = M::InferenceState>,
    H: RecurrentPPOHook<M>,
{
    type Tensor = M::InferenceTensor;
    type RolloutState = M::InferenceState;
    type Actor = M::InferencePolicy;

    fn actor(&self) -> Self::Actor {
        self.lm.inference_policy()
    }

    fn learn<B: TrajectoryBatch<Self::Tensor, State = Self::RolloutState>>(
        &mut self,
        batches: &[B],
    ) -> Result<()> {
        RecurrentPPO::learn(self, batches)
    }
}
