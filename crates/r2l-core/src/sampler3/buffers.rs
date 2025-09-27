use crate::distributions::Policy;
use crate::tensor::R2lTensor;
use crate::utils::rollout_buffer::{Advantages, Logps, Returns};
use crate::{
    env::{Env, Memory},
    policies::ValueFunction,
    rng::RNG,
    sampler2::{Buffer, CollectionBound},
};
use rand::seq::SliceRandom;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::{Arc, Mutex, MutexGuard},
};

pub struct FixedSizeStateBuffer<E: Env> {
    capacity: usize,
    pub states: AllocRingBuffer<E::Tensor>,
    pub next_states: AllocRingBuffer<E::Tensor>,
    pub rewards: AllocRingBuffer<f32>,
    pub action: AllocRingBuffer<E::Tensor>,
    pub terminated: AllocRingBuffer<bool>,
    pub trancuated: AllocRingBuffer<bool>,
}

impl<E: Env> FixedSizeStateBuffer<E> {
    pub fn push(
        &mut self,
        state: E::Tensor,
        next_state: E::Tensor,
        action: E::Tensor,
        reward: f32,
        terminated: bool,
        trancuated: bool,
    ) {
        self.states.enqueue(state);
        self.next_states.enqueue(next_state);
        self.action.enqueue(action);
        self.rewards.enqueue(reward);
        self.terminated.enqueue(terminated);
        self.trancuated.enqueue(trancuated);
    }
}

impl<E: Env> Buffer for FixedSizeStateBuffer<E> {
    type Tensor = <E as Env>::Tensor;

    fn states(&self) -> Vec<Self::Tensor> {
        self.states.iter().cloned().collect()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        self.next_states.iter().cloned().collect()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        self.action.iter().cloned().collect()
    }

    fn rewards(&self) -> Vec<f32> {
        self.rewards.iter().cloned().collect()
    }

    fn terminated(&self) -> Vec<bool> {
        self.terminated.iter().cloned().collect()
    }

    fn trancuated(&self) -> Vec<bool> {
        self.trancuated.iter().cloned().collect()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        } = snapshot;
        self.push(state, next_state, action, reward, terminated, trancuated);
    }

    fn last_state_terminates(&self) -> bool {
        *self.terminated.back().unwrap() || *self.trancuated.back().unwrap()
    }

    fn build(collection_bound: CollectionBound) -> Self {
        todo!()
    }
}

pub struct VariableSizedStateBuffer<E: Env> {
    pub states: Vec<E::Tensor>,
    pub next_states: Vec<E::Tensor>,
    pub rewards: Vec<f32>,
    pub action: Vec<E::Tensor>,
    pub terminated: Vec<bool>,
    pub trancuated: Vec<bool>,
}

// TODO: for some reason the Default proc macro trips up the compiler. We should investigate this
// in the future.
impl<E: Env> Default for VariableSizedStateBuffer<E> {
    fn default() -> Self {
        Self {
            states: vec![],
            next_states: vec![],
            rewards: vec![],
            action: vec![],
            terminated: vec![],
            trancuated: vec![],
        }
    }
}

impl<E: Env> VariableSizedStateBuffer<E> {
    pub fn push(
        &mut self,
        state: E::Tensor,
        next_state: E::Tensor,
        action: E::Tensor,
        reward: f32,
        terminated: bool,
        trancuated: bool,
    ) {
        self.states.push(state);
        self.next_states.push(next_state);
        self.action.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.trancuated.push(trancuated);
    }
}

impl<E: Env> Buffer for VariableSizedStateBuffer<E> {
    type Tensor = <E as Env>::Tensor;

    fn states(&self) -> Vec<Self::Tensor> {
        self.states.iter().cloned().collect()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        self.next_states.iter().cloned().collect()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        self.action.iter().cloned().collect()
    }

    fn rewards(&self) -> Vec<f32> {
        self.rewards.clone()
    }

    fn terminated(&self) -> Vec<bool> {
        todo!()
    }

    fn trancuated(&self) -> Vec<bool> {
        todo!()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        let Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        } = snapshot;
        self.push(state, next_state, action, reward, terminated, trancuated);
    }

    fn last_state_terminates(&self) -> bool {
        *self.terminated.last().unwrap() || *self.trancuated.last().unwrap()
    }

    fn build(collection_bound: CollectionBound) -> Self {
        todo!()
    }
}

pub struct RcBufferWrapper<B: Buffer>(pub Rc<RefCell<B>>);

impl<'a, B: Buffer> RcBufferWrapper<B> {
    pub fn new(buffer: B) -> Self {
        Self(Rc::new(RefCell::new(buffer)))
    }

    pub fn build(collection_bound: CollectionBound) -> Self {
        Self(Rc::new(RefCell::new(B::build(collection_bound))))
    }

    pub fn buffer(&'a self) -> Ref<'a, B> {
        self.0.borrow()
    }

    pub fn buffer_mut(&'a self) -> RefMut<'a, B> {
        self.0.borrow_mut()
    }
}

impl<B: Buffer> Clone for RcBufferWrapper<B> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[derive(Debug)]
pub struct ArcBufferWrapper<B: Buffer>(pub Arc<Mutex<B>>);

impl<'a, B: Buffer> ArcBufferWrapper<B> {
    pub fn new(buffer: Arc<Mutex<B>>) -> Self {
        Self(buffer)
    }

    pub fn buffer(&'a self) -> MutexGuard<'a, B> {
        self.0.lock().unwrap()
    }
}

impl<B: Buffer> Clone for ArcBufferWrapper<B> {
    fn clone(&self) -> Self {
        ArcBufferWrapper(self.0.clone())
    }
}

pub struct BatchIndexIterator {
    indicies: Vec<(usize, usize)>,
    sample_size: usize,
    current: usize,
}

impl BatchIndexIterator {
    pub fn new<B: Buffer, BT: Deref<Target = B>>(buffers: &[BT], sample_size: usize) -> Self {
        let mut indicies = (0..buffers.len())
            .flat_map(|i| {
                let rb = &buffers[i];
                (0..rb.total_steps()).map(|j| (i, j)).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        RNG.with_borrow_mut(|rng| indicies.shuffle(rng));
        Self {
            indicies,
            sample_size,
            current: 0,
        }
    }

    pub fn iter(&mut self) -> Option<Vec<(usize, usize)>> {
        let total_size = self.indicies.len();
        if self.sample_size + self.current >= total_size {
            return None;
        }
        let batch_indicies = &self.indicies[self.current..self.current + self.sample_size];
        self.current += self.sample_size;
        Some(batch_indicies.to_owned())
    }
}

fn calculate_advantages_and_returns<
    B: Buffer,
    VT: R2lTensor + From<B::Tensor>,
    BT: Deref<Target = B>,
>(
    buffers: &[BT],
    value_func: &impl ValueFunction<Tensor = VT>,
    gamma: f32,
    lambda: f32,
) -> (Advantages, Returns) {
    let mut advantage_vec = vec![];
    let mut returns_vec = vec![];

    for buff in buffers {
        let total_steps = buff.total_steps();
        let mut all_states: Vec<VT> = buff.convert_states();
        all_states.push(buff.last_state().unwrap().into());

        let values_stacked = value_func.calculate_values(&all_states).unwrap();
        let values = values_stacked.to_vec();
        let mut advantages: Vec<f32> = vec![0.; total_steps];
        let mut returns: Vec<f32> = vec![0.; total_steps];
        let mut last_gae_lam: f32 = 0.;

        for i in (0..total_steps).rev() {
            let next_non_terminal = if buff.dones()[i] {
                last_gae_lam = 0.;
                0f32
            } else {
                1.
            };
            let delta = buff.rewards()[i] + next_non_terminal * gamma * values[i + 1] - values[i];
            last_gae_lam = delta + next_non_terminal * gamma * lambda * last_gae_lam;
            advantages[i] = last_gae_lam;
            returns[i] = last_gae_lam + values[i];
        }
        advantage_vec.push(advantages);
        returns_vec.push(returns);
    }

    (Advantages(advantage_vec), Returns(returns_vec))
}

// TODO: whether this is the right construct or not remains to be seen
pub enum BufferStack<B: Buffer> {
    RefCounted(Vec<RcBufferWrapper<B>>),
    AtomicRefCounted(Vec<ArcBufferWrapper<B>>),
}

// TODO: we may want to not repeat ourselfs so much here!
impl<B: Buffer> BufferStack<B> {
    pub fn advantages_and_returns<VT: R2lTensor + From<B::Tensor>>(
        &self,
        value_func: &impl ValueFunction<Tensor = VT>,
        gamma: f32,
        lambda: f32,
    ) -> (Advantages, Returns) {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                calculate_advantages_and_returns(&buffers, value_func, gamma, lambda)
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                calculate_advantages_and_returns(&buffers, value_func, gamma, lambda)
            }
        }
    }

    pub fn sample<VT: R2lTensor + From<B::Tensor>>(
        &self,
        indicies: &[(usize, usize)],
    ) -> (Vec<VT>, Vec<VT>) {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                let mut observations = vec![];
                let mut actions = vec![];
                for (buffer_idx, idx) in indicies {
                    let observation = buffers[*buffer_idx].convert_states::<VT>()[*idx].clone();
                    let action = buffers[*buffer_idx].convert_actions::<VT>()[*idx].clone();
                    observations.push(observation.clone());
                    actions.push(action.clone());
                }
                (observations, actions)
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                let mut observations = vec![];
                let mut actions = vec![];
                for (buffer_idx, idx) in indicies {
                    let observation = buffers[*buffer_idx].convert_states::<VT>()[*idx].clone();
                    let action = buffers[*buffer_idx].convert_actions::<VT>()[*idx].clone();
                    observations.push(observation.clone());
                    actions.push(action.clone());
                }
                (observations, actions)
            }
        }
    }

    pub fn index_iterator(&self, sample_size: usize) -> BatchIndexIterator {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                BatchIndexIterator::new(&buffers, sample_size)
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                BatchIndexIterator::new(&buffers, sample_size)
            }
        }
    }

    pub fn states<T: From<B::Tensor>>(&self, idx: usize) -> Vec<T> {
        match self {
            Self::RefCounted(buffers) => buffers[idx].buffer().convert_states(),
            Self::AtomicRefCounted(buffers) => buffers[idx].buffer().convert_states(),
        }
    }

    pub fn next_states<T: From<B::Tensor>>(&self, idx: usize) -> Vec<T> {
        match self {
            Self::RefCounted(buffers) => buffers[idx].buffer().convert_next_states(),
            Self::AtomicRefCounted(buffers) => buffers[idx].buffer().convert_next_states(),
        }
    }

    pub fn actions<T: From<B::Tensor>>(&self, idx: usize) -> Vec<T> {
        match self {
            Self::RefCounted(buffers) => buffers[idx].buffer().convert_actions(),
            Self::AtomicRefCounted(buffers) => buffers[idx].buffer().convert_actions(),
        }
    }

    pub fn logps<PT: R2lTensor + From<B::Tensor>>(
        &self,
        policy: &impl Policy<Tensor = PT>,
    ) -> Logps {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                let mut logps = vec![];
                for buff in &buffers {
                    let states = buff.convert_states();
                    let actions = buff.convert_actions();
                    let logp = policy
                        .log_probs(&states, &actions)
                        .map(|t| t.to_vec())
                        .unwrap();
                    logps.push(logp);
                }
                Logps(logps)
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                let mut logps = vec![];
                for buff in &buffers {
                    let states = buff.convert_states();
                    let actions = buff.convert_actions();
                    let logp = policy
                        .log_probs(&states, &actions)
                        .map(|t| t.to_vec())
                        .unwrap();
                    logps.push(logp);
                }
                Logps(logps)
            }
        }
    }

    pub fn total_rewards(&self) -> f32 {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                buffers
                    .iter()
                    .map(|s| s.rewards().iter().sum::<f32>())
                    .sum::<f32>()
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                buffers
                    .iter()
                    .map(|s| s.rewards().iter().sum::<f32>())
                    .sum::<f32>()
            }
        }
    }

    pub fn total_episodes(&self) -> usize {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                buffers
                    .iter()
                    .flat_map(|s| s.dones())
                    .filter(|d| *d)
                    .count()
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                buffers
                    .iter()
                    .flat_map(|s| s.dones())
                    .filter(|d| *d)
                    .count()
            }
        }
    }

    pub fn total_steps(&self) -> usize {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                buffers.iter().map(|b| b.states().len()).sum()
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                buffers.iter().map(|b| b.states().len()).sum()
            }
        }
    }
}
