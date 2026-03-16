use crate::{
    Algorithm,
    agents::{Agent, Agent5, Agent6, TensorOfAgent},
    distributions::Policy,
    env::{Env, Sampler, TensorOfSampler},
    sampler::PolicyWrapper,
    sampler4::{
        Sampler3, Sampler4,
        buffer::{BufferS, BufferS2, TrajectoryContainer, buffer_wrapper},
    },
    tensor::R2lTensor,
    utils::rollout_buffer::RolloutBuffer,
};
use anyhow::Result;
use burn::tensor::T;
use std::marker::PhantomData;

macro_rules! break_on_hook_res {
    ($hook_res:expr) => {
        if $hook_res {
            break;
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub enum LearningSchedule {
    RolloutBound {
        total_rollouts: usize,
        current_rollout: usize,
    },
    TotalStepBound {
        total_steps: usize,
        current_step: usize,
    },
}

impl LearningSchedule {
    pub fn total_step_bound(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps,
            current_step: 0,
        }
    }
}

pub trait OnPolicyAlgorithmHooks<T: Clone> {
    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(&mut self, rollouts: &mut [RolloutBuffer<T>]) -> bool;

    fn post_training_hook(&mut self) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
}

pub struct DefaultOnPolicyAlgorightmsHooks {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
}

impl DefaultOnPolicyAlgorightmsHooks {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
        }
    }
}

impl<T: Clone> OnPolicyAlgorithmHooks<T> for DefaultOnPolicyAlgorightmsHooks {
    fn init_hook(&mut self) -> bool {
        false
    }

    fn post_rollout_hook(&mut self, rollouts: &mut [RolloutBuffer<T>]) -> bool {
        let total_reward = rollouts
            .iter()
            .map(|s| s.rewards.iter().sum::<f32>())
            .sum::<f32>();
        let episodes: usize = rollouts
            .iter()
            .flat_map(|s| &s.dones)
            .filter(|d| **d)
            .count();
        println!(
            "rollout: {:<3} episodes: {:<5} total reward: {:<5.2} avg reward per episode: {:.2}",
            self.rollout_idx,
            episodes,
            total_reward,
            total_reward / episodes as f32
        );
        self.rollout_idx += 1;
        match &mut self.learning_schedule {
            LearningSchedule::RolloutBound {
                total_rollouts,
                current_rollout,
            } => {
                *current_rollout += 1;
                current_rollout >= total_rollouts
            }
            LearningSchedule::TotalStepBound {
                total_steps,
                current_step,
            } => {
                let rollout_steps: usize = rollouts.iter().map(|e| e.actions.len()).sum();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        }
    }

    fn post_training_hook(&mut self) -> bool {
        false
    }

    fn shutdown_hook(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct OnPolicyAlgorithm<S: Sampler, A: Agent, H: OnPolicyAlgorithmHooks<TensorOfAgent<A>>> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

impl<S: Sampler, A: Agent, H: OnPolicyAlgorithmHooks<TensorOfAgent<A>>> Algorithm
    for OnPolicyAlgorithm<S, A, H>
where
    TensorOfSampler<S>: From<TensorOfAgent<A>>,
    TensorOfSampler<S>: Into<TensorOfAgent<A>>,
    <A as Agent>::Policy: Clone,
{
    fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            // rollout phase
            let distribution = self.agent.policy();
            let mut rollouts = self.sampler.collect_rollouts(distribution)?;
            break_on_hook_res!(self.hooks.post_rollout_hook(&mut rollouts));

            // learning phase
            self.agent.learn(rollouts)?;
            break_on_hook_res!(self.hooks.post_training_hook());
        }
        self.hooks.shutdown_hook()
    }
}

pub struct DefaultOnPolicyAlgorightmsHooks4<E: Env, P: Policy> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _buffer: PhantomData<E>,
    _policy: PhantomData<P>,
}

impl<E: Env, P: Policy> DefaultOnPolicyAlgorightmsHooks4<E, P> {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _buffer: PhantomData,
            _policy: PhantomData,
        }
    }
}

pub trait OnPolicyAlgorithmHooks5
where
    Self::P: Clone,
    <Self::P as Policy>::Tensor: From<<Self::E as Env>::Tensor>,
    <Self::E as Env>::Tensor: From<<Self::P as Policy>::Tensor>,
{
    type P: Policy;
    type E: Env;

    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook<C: TrajectoryContainer>(&mut self, rollouts: &[BufferS<C>]) -> bool;

    fn post_training_hook(&mut self, policy: Self::P) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
}

pub struct OnPolicyAlgorithm5<
    H: OnPolicyAlgorithmHooks5,
    A: Agent5<Policy = H::P>,
    S: Sampler3<Env = H::E>,
> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

impl<H: OnPolicyAlgorithmHooks5, A: Agent5<Policy = H::P>, S: Sampler3<Env = H::E>>
    OnPolicyAlgorithm5<H, A, S>
{
    pub fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            let policy = self.agent.policy3();
            let buffers = self.sampler.collect_rollouts(PolicyWrapper::new(policy));
            break_on_hook_res!(self.hooks.post_rollout_hook(buffers.as_ref()));

            // learning phase
            self.agent.learn4(buffers.as_ref())?;
            let policy = self.agent.policy3();
            break_on_hook_res!(self.hooks.post_training_hook(policy));
        }
        self.hooks.shutdown_hook()
    }
}

pub trait OnPolicyAlgorithmHooks6 {
    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook<T: R2lTensor>(&mut self, rollouts: &[BufferS2<T>]) -> bool;

    fn post_training_hook(&mut self, policy: impl Policy) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
}

pub struct OnPolicyAlgorithm6<H: OnPolicyAlgorithmHooks6, A: Agent6, S: Sampler4> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

// That is the final version! This is kinda ok
impl<H: OnPolicyAlgorithmHooks6, A: Agent6, S: Sampler4> OnPolicyAlgorithm6<H, A, S>
where
    S::Tensor: From<A::Tensor>,
    A::Tensor: From<S::Tensor>,
    A::Policy: Clone,
{
    pub fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            let policy = self.agent.policy3();
            let policy = PolicyWrapper::new(policy);
            let buffers = self.sampler.collect_rollouts(policy);
            let buffers = buffer_wrapper(&buffers);
            break_on_hook_res!(self.hooks.post_rollout_hook(buffers.as_ref()));

            self.agent.learn4(buffers.as_ref())?;
            let policy = self.agent.policy3();
            break_on_hook_res!(self.hooks.post_training_hook(policy));
        }
        self.hooks.shutdown_hook()
    }
}
