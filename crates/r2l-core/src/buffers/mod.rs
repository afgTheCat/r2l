pub mod fix_sized;
pub mod variable_sized;

use itertools::izip;

use crate::tensor::R2lTensor;

/// One transition collected from an environment.
#[derive(Debug)]
pub struct Memory<T> {
    /// Observation before the action.
    pub state: T,
    /// Observation after the action.
    pub next_state: T,
    /// Action selected by the actor.
    pub action: T,
    /// Reward emitted by the environment.
    pub reward: f32,
    /// Whether the transition ended in a terminal state.
    pub terminated: bool,
    /// Whether the transition ended because of a time limit or external cutoff.
    pub truncated: bool,
}

impl<T> Memory<T> {
    /// Returns `true` when the transition ends the episode for any reason.
    pub fn is_done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Read-only access to a trajectory or rollout buffer.
///
/// All iterators should yield items in the same order and length. Algorithms
/// assume that the `n`th state, action, reward, and done flags describe one
/// transition.
pub trait TrajectoryContainer: Sync {
    /// Tensor type stored in this trajectory.
    type Tensor: R2lTensor;

    /// Number of transitions in the trajectory.
    fn len(&self) -> usize;

    /// Returns whether the trajectory contains no transitions.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over observations before each action.
    fn states(&self) -> impl Iterator<Item = &Self::Tensor>;

    /// Iterates over observations after each action.
    fn next_states(&self) -> impl Iterator<Item = &Self::Tensor>;

    /// Iterates over actions.
    fn actions(&self) -> impl Iterator<Item = &Self::Tensor>;

    /// Iterates over rewards.
    fn rewards(&self) -> impl Iterator<Item = f32>;

    /// Iterates over terminal-state flags.
    fn terminated(&self) -> impl Iterator<Item = bool>;

    /// Iterates over time-limit or external-cutoff flags.
    fn truncated(&self) -> impl Iterator<Item = bool>;

    /// Iterates over combined done flags.
    fn dones(&self) -> impl Iterator<Item = bool> {
        self.terminated()
            .zip(self.truncated())
            .map(|(terminated, truncated)| terminated || truncated)
    }

    /// Iterates over cloned transitions.
    ///
    /// This is convenient for conversion code but may be expensive for large
    /// tensor payloads.
    fn memories(&self) -> impl Iterator<Item = Memory<Self::Tensor>> {
        izip!(
            self.states(),
            self.next_states(),
            self.actions(),
            self.rewards(),
            self.terminated(),
            self.truncated()
        )
        .map(
            |(state, next_state, action, reward, terminated, truncated)| Memory {
                state: state.clone(),
                next_state: next_state.clone(),
                action: action.clone(),
                reward,
                terminated,
                truncated,
            },
        )
    }
}

/// Mutable trajectory buffer that can receive newly collected transitions.
pub trait ExpandableTrajectoryContainer: TrajectoryContainer + Send + 'static {
    /// Appends one transition to the buffer.
    fn push(&mut self, memory: Memory<Self::Tensor>);
}

/// Mutable access for correcting the most recent state or reward.
pub trait EditableTrajectoryContainer: TrajectoryContainer {
    /// Removes and returns the last stored state.
    fn pop_last_state(&mut self) -> Self::Tensor;

    /// Removes and returns the last stored reward.
    fn pop_last_reward(&mut self) -> f32;

    /// Replaces the last stored state.
    fn set_last_state(&mut self, t: Self::Tensor);

    /// Replaces the last stored reward.
    fn set_last_reward(&mut self, r: f32);
}
