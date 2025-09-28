use crate::{
    Algorithm,
    agents::{Agent, Agent3, Agent4, TensorOfAgent},
    distributions::Policy,
    env::{Env, Sampler, Sampler3, Sampler4, TensorOfSampler},
    sampler::PolicyWrapper,
    sampler3::{
        buffer_stack::BufferStack3,
        buffers::{Buffer, BufferStack},
    },
    utils::rollout_buffer::RolloutBuffer,
};
use anyhow::Result;
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

pub trait OnPolicyAlgorithmHooks2 {
    type B: Buffer;
    type P: Policy;

    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(&mut self, rollouts: &[Self::B]) -> bool;

    fn post_training_hook(&mut self, policy: Self::P) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
}

pub struct DefaultOnPolicyAlgorightmsHooks2<B: Buffer, P: Policy> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _buffer: PhantomData<B>,
    _policy: PhantomData<P>,
}

impl<B: Buffer, P: Policy> DefaultOnPolicyAlgorightmsHooks2<B, P> {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _buffer: PhantomData,
            _policy: PhantomData,
        }
    }
}

impl<B: Buffer, P: Policy> OnPolicyAlgorithmHooks2 for DefaultOnPolicyAlgorightmsHooks2<B, P> {
    type B = B;
    type P = P;

    fn init_hook(&mut self) -> bool {
        false
    }

    fn post_rollout_hook(&mut self, rollouts: &[B]) -> bool {
        let total_reward = rollouts
            .iter()
            .map(|s| s.rewards().iter().sum::<f32>())
            .sum::<f32>();
        let episodes: usize = rollouts
            .iter()
            .flat_map(|s| s.dones())
            .filter(|d| *d)
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
                let rollout_steps: usize = rollouts.iter().map(|e| e.actions().len()).sum();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        }
    }

    fn post_training_hook(&mut self, _policy: P) -> bool {
        false
    }

    fn shutdown_hook(&mut self) -> Result<()> {
        Ok(())
    }
}

pub trait OnPolicyAlgorithmHooks3
where
    Self::P: Clone,
    <Self::P as Policy>::Tensor: From<<Self::B as Buffer>::Tensor>,
    <Self::B as Buffer>::Tensor: From<<Self::P as Policy>::Tensor>,
{
    type B: Buffer;
    type P: Policy;

    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(&mut self, rollouts: &BufferStack<Self::B>) -> bool;

    fn post_training_hook(&mut self, policy: Self::P) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
}

pub struct DefaultOnPolicyAlgorightmsHooks3<B: Buffer, P: Policy> {
    rollout_idx: usize,
    learning_schedule: LearningSchedule,
    _buffer: PhantomData<B>,
    _policy: PhantomData<P>,
}

impl<B: Buffer, P: Policy> DefaultOnPolicyAlgorightmsHooks3<B, P> {
    pub fn new(learning_schedule: LearningSchedule) -> Self {
        Self {
            rollout_idx: 0,
            learning_schedule,
            _buffer: PhantomData,
            _policy: PhantomData,
        }
    }
}

impl<B: Buffer, P: Policy + Clone> OnPolicyAlgorithmHooks3
    for DefaultOnPolicyAlgorightmsHooks3<B, P>
where
    P::Tensor: From<B::Tensor>,
    B::Tensor: From<P::Tensor>,
{
    type B = B;
    type P = P;

    fn init_hook(&mut self) -> bool {
        false
    }

    fn post_rollout_hook(&mut self, rollouts: &BufferStack<B>) -> bool {
        let total_reward = rollouts.total_rewards();
        let episodes = rollouts.total_episodes();
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
                let rollout_steps: usize = rollouts.total_steps();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        }
    }

    fn post_training_hook(&mut self, _policy: P) -> bool {
        false
    }

    fn shutdown_hook(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct OnPolicyAlgorithm3<
    H: OnPolicyAlgorithmHooks3,
    S: Sampler3<Buffer = H::B>,
    A: Agent3<Policy = H::P>,
> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

impl<H: OnPolicyAlgorithmHooks3, S: Sampler3<Buffer = H::B>, A: Agent3<Policy = H::P>>
    OnPolicyAlgorithm3<H, S, A>
{
    pub fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            let policy = self.agent.policy3();
            let buffers = self.sampler.collect_rollouts(policy);
            break_on_hook_res!(self.hooks.post_rollout_hook(&buffers));

            // learning phase
            self.agent.learn3(buffers)?;
            let policy = self.agent.policy3();
            break_on_hook_res!(self.hooks.post_training_hook(policy));
        }
        self.hooks.shutdown_hook()
    }
}

pub trait OnPolicyAlgorithmHooks4
where
    Self::P: Clone,
    <Self::P as Policy>::Tensor: From<<Self::E as Env>::Tensor>,
    <Self::E as Env>::Tensor: From<<Self::P as Policy>::Tensor>,
{
    type P: Policy;
    type E: Env;

    fn init_hook(&mut self) -> bool;

    fn post_rollout_hook(&mut self, rollouts: &BufferStack3<<Self::P as Policy>::Tensor>) -> bool;

    fn post_training_hook(&mut self, policy: Self::P) -> bool;

    fn shutdown_hook(&mut self) -> Result<()>;
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

impl<E: Env, P: Policy + Clone> OnPolicyAlgorithmHooks4 for DefaultOnPolicyAlgorightmsHooks4<E, P>
where
    P::Tensor: From<E::Tensor>,
    E::Tensor: From<P::Tensor>,
    P: Clone,
{
    type E = E;
    type P = P;

    fn init_hook(&mut self) -> bool {
        false
    }

    fn post_rollout_hook(&mut self, rollouts: &BufferStack3<<Self::P as Policy>::Tensor>) -> bool {
        let total_reward = rollouts.total_rewards();
        let episodes = rollouts.total_episodes();
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
                let rollout_steps: usize = rollouts.total_steps();
                *current_step += rollout_steps;
                current_step >= total_steps
            }
        }
    }

    fn post_training_hook(&mut self, _policy: P) -> bool {
        false
    }

    fn shutdown_hook(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct OnPolicyAlgorithm4<
    H: OnPolicyAlgorithmHooks4,
    A: Agent4<Policy = H::P>,
    S: Sampler4<Env = H::E>,
> {
    pub sampler: S,
    pub agent: A,
    pub hooks: H,
}

impl<H: OnPolicyAlgorithmHooks4, A: Agent4<Policy = H::P>, S: Sampler4<Env = H::E>>
    OnPolicyAlgorithm4<H, A, S>
{
    pub fn train(&mut self) -> Result<()> {
        if self.hooks.init_hook() {
            return Ok(());
        }
        loop {
            let policy = self.agent.policy3();
            let policy = PolicyWrapper::new(policy);
            self.sampler.collect_rollouts(policy);
            let buffers = self.sampler.get_buffer_stack();
            break_on_hook_res!(self.hooks.post_rollout_hook(&buffers));

            // learning phase
            self.agent.learn3(buffers)?;
            let policy = self.agent.policy3();
            break_on_hook_res!(self.hooks.post_training_hook(policy));
        }
        self.hooks.shutdown_hook()
    }
}
