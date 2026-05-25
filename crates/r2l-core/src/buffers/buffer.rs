use crate::{
    buffers::{
        Memory,
        // reusable_vec::{ReusableVec, ReusableVecSlice},
    },
    tensor::R2lTensor,
};

// pub enum TrajectoryTensorField<'a, T: R2lTensor> {
//     // TODO: we cold probably use Rc<T> here?
//     Owned(Vec<T>),
//     Borrowed(ReusableVecSlice<'a, T>),
// }
//
// impl<'a, T: R2lTensor> AsRef<[T]> for TrajectoryTensorField<'a, T> {
//     fn as_ref(&self) -> &[T] {
//         match self {
//             Self::Owned(data) => data.as_slice(),
//             Self::Borrowed(data) => data.as_ref(),
//         }
//     }
// }

// pub struct TrajectoryBatch<'a, T: R2lTensor> {
//     states: TrajectoryTensorField<'a, T>,
//     next_states: TrajectoryTensorField<'a, T>,
//     actions: TrajectoryTensorField<'a, T>,
//     rewards: ReusableVecSlice<'a, f32>,
//     terminated: ReusableVecSlice<'a, bool>,
//     truncated: ReusableVecSlice<'a, bool>,
// }
//
// impl<'a, T: R2lTensor> TrajectoryBatch<'a, T> {
//     pub fn len(&self) -> usize {
//         self.states.as_ref().len()
//     }
//
//     pub fn is_empty(&self) -> bool {
//         self.len() == 0
//     }
//
//     pub fn states(&self) -> &[T] {
//         self.states.as_ref()
//     }
//
//     pub fn next_states(&self) -> &[T] {
//         self.next_states.as_ref()
//     }
//
//     pub fn actions(&self) -> &[T] {
//         self.actions.as_ref()
//     }
//
//     pub fn rewards(&self) -> &[f32] {
//         self.rewards.as_ref()
//     }
//
//     pub fn terminated(&self) -> &[bool] {
//         self.terminated.as_ref()
//     }
//
//     pub fn truncated(&self) -> &[bool] {
//         self.truncated.as_ref()
//     }
// }

// the new buffer type I am experimenting with. Probably going to make things faster
#[derive(Clone)]
pub struct NewBuffer<T: R2lTensor> {
    states: Vec<T>,
    next_states: Vec<T>,
    actions: Vec<T>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
}

impl<T: R2lTensor> Default for NewBuffer<T> {
    fn default() -> Self {
        Self {
            states: Default::default(),
            next_states: Default::default(),
            actions: Default::default(),
            rewards: Default::default(),
            terminated: Default::default(),
            truncated: Default::default(),
        }
    }
}

pub struct TrajectoryView<'a, T: R2lTensor> {
    pub states: &'a [T],
    pub next_states: &'a [T],
    pub actions: &'a [T],
    pub rewards: &'a [f32],
    pub terminated: &'a [bool],
    pub truncated: &'a [bool],
}

impl<'a, T: R2lTensor> TrajectoryView<'a, T> {
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

impl<T: R2lTensor> NewBuffer<T> {
    pub fn clear(&mut self) {
        self.states.clear();
        self.next_states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.terminated.clear();
        self.truncated.clear();
    }

    pub fn push(&mut self, memory: Memory<T>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            truncated,
        } = memory;
        self.states.push(state);
        self.next_states.push(next_state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.truncated.push(truncated);
    }

    pub fn to_trajectory_view(&self) -> TrajectoryView<'_, T> {
        TrajectoryView {
            states: &self.states,
            next_states: &self.next_states,
            actions: &self.actions,
            rewards: &self.rewards,
            terminated: &self.terminated,
            truncated: &self.truncated,
        }
    }
}
