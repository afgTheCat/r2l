use crate::{
    buffers::{Memory, TrajectoryBatch},
    tensor::R2lTensor,
};

// the new buffer type I am experimenting with. Probably going to make things faster
#[derive(Clone)]
pub struct TrajectoryBuffer<T: R2lTensor, S: Clone + Send + Sync + 'static = ()> {
    states: Vec<T>,
    next_states: Vec<T>,
    actions: Vec<T>,
    actor_states: Vec<Option<S>>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl<T: R2lTensor, S: Clone + Send + Sync + 'static> Default for TrajectoryBuffer<T, S> {
    fn default() -> Self {
        Self {
            states: Default::default(),
            next_states: Default::default(),
            actions: Default::default(),
            actor_states: Default::default(),
            rewards: Default::default(),
            terminated: Default::default(),
            truncated: Default::default(),
        }
    }
}

pub struct TrajectoryView<'a, T: R2lTensor, S: Clone + Send + Sync + 'static = ()> {
    pub states: &'a [T],
    pub next_states: &'a [T],
    pub actions: &'a [T],
    pub actor_states: &'a [Option<S>],
    pub rewards: &'a [f32],
    pub terminated: &'a [bool],
    pub truncated: &'a [bool],
}

impl<'a, T: R2lTensor, S: Clone + Send + Sync + 'static> TrajectoryBatch<T>
    for TrajectoryView<'a, T, S>
{
    type State = S;

    fn len(&self) -> usize {
        self.states.len()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn states(&self) -> &[T] {
        self.states
    }

    fn next_states(&self) -> &[T] {
        self.next_states
    }

    fn actions(&self) -> &[T] {
        self.actions
    }

    fn actor_states(&self) -> &[Option<Self::State>] {
        self.actor_states
    }

    fn rewards(&self) -> &[f32] {
        self.rewards
    }

    fn terminated(&self) -> &[bool] {
        self.terminated
    }

    fn truncated(&self) -> &[bool] {
        self.truncated
    }
}

impl<'a, T: R2lTensor, S: Clone + Send + Sync + 'static> TrajectoryView<'a, T, S> {
    pub fn dones(&self) -> impl Iterator<Item = bool> {
        self.terminated
            .iter()
            .zip(self.truncated.iter())
            .map(|(terminated, truncated)| *terminated || *truncated)
    }

    pub fn episode_terminations(&self) -> usize {
        self.dones().filter(|x| *x).count()
    }
}

impl<T: R2lTensor, S: Clone + Send + Sync + 'static> TrajectoryBuffer<T, S> {
    pub fn clear(&mut self) {
        self.states.clear();
        self.next_states.clear();
        self.actions.clear();
        self.actor_states.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
    }

    pub fn push(&mut self, memory: Memory<T, S>) {
        let Memory {
            state,
            next_state,
            action,
            actor_state,
            reward,
            terminated,
            truncated,
        } = memory;
        self.states.push(state);
        self.next_states.push(next_state);
        self.actions.push(action);
        self.actor_states.push(actor_state);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.truncated.push(truncated);
    }

    pub fn replace_last_next_state(&mut self, next_state: T) {
        if let Some(last_next_state) = self.next_states.last_mut() {
            *last_next_state = next_state;
        }
    }

    pub fn to_trajectory_view(&self) -> TrajectoryView<'_, T, S> {
        TrajectoryView {
            states: &self.states,
            next_states: &self.next_states,
            actions: &self.actions,
            actor_states: &self.actor_states,
            rewards: &self.rewards,
            terminated: &self.terminated,
            truncated: &self.truncated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TrajectoryBuffer;
    use crate::{
        buffers::{Memory, TrajectoryBatch},
        tensor::TensorData,
    };

    #[test]
    fn actor_states_stay_aligned_with_transitions() {
        let mut buffer = TrajectoryBuffer::<TensorData, usize>::default();
        buffer.push(Memory {
            state: TensorData::from_vec(vec![0.0]),
            next_state: TensorData::from_vec(vec![1.0]),
            action: TensorData::from_vec(vec![1.0]),
            actor_state: Some(7),
            reward: 1.0,
            terminated: false,
            truncated: false,
        });

        let view = buffer.to_trajectory_view();
        assert_eq!(view.len(), 1);
        assert_eq!(view.actor_states(), &[Some(7)]);
    }
}
