use crate::distributions::Policy;
use crate::env::SnapShot;
use crate::sampler3::CollectionBound;
use crate::tensor::R2lTensor;
use crate::utils::rollout_buffer::{Advantages, Logps, Returns};
use crate::{
    env::{Env, Memory},
    policies::ValueFunction,
    rng::RNG,
};
use rand::Rng;
use rand::seq::SliceRandom;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::{Arc, Mutex, MutexGuard},
};

pub trait Buffer: Sized {
    type Tensor: R2lTensor;

    fn states(&self) -> Vec<Self::Tensor>;

    fn convert_states<T: From<Self::Tensor>>(&self) -> Vec<T> {
        self.states().into_iter().map(|t| t.into()).collect()
    }

    fn next_states(&self) -> Vec<Self::Tensor>;

    fn convert_next_states<T: From<Self::Tensor>>(&self) -> Vec<T> {
        self.next_states().into_iter().map(|t| t.into()).collect()
    }

    fn actions(&self) -> Vec<Self::Tensor>;

    fn convert_actions<T: From<Self::Tensor>>(&self) -> Vec<T> {
        self.actions().into_iter().map(|t| t.into()).collect()
    }

    fn rewards(&self) -> Vec<f32>;

    fn terminated(&self) -> Vec<bool>;

    fn trancuated(&self) -> Vec<bool>;

    fn push(&mut self, snapshot: Memory<Self::Tensor>);

    fn dones(&self) -> Vec<bool> {
        self.terminated()
            .into_iter()
            .zip(self.trancuated().into_iter())
            .map(|(terminated, trancuated)| terminated || trancuated)
            .collect()
    }

    fn total_steps(&self) -> usize;

    fn last_state(&self) -> Option<Self::Tensor>;

    fn last_state_terminates(&self) -> bool;

    #[inline(always)]
    fn step<E: Env<Tensor = Self::Tensor>>(
        &mut self,
        env: &mut E,
        distr: &Box<dyn Policy<Tensor = Self::Tensor>>,
        last_state: Option<Self::Tensor>,
    ) {
        let state = if let Some(state) = self.last_state() {
            state
        } else if let Some(last_state) = last_state {
            last_state
        } else {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            env.reset(seed).unwrap()
        };
        let action = distr.get_action(state.clone()).unwrap();
        let SnapShot {
            state: mut next_state,
            reward,
            terminated,
            trancuated,
        } = env.step(action.clone()).unwrap();
        let done = terminated || trancuated;
        if done {
            let seed = RNG.with_borrow_mut(|rng| rng.random::<u64>());
            next_state = env.reset(seed).unwrap();
        }
        self.push(Memory {
            state,
            next_state,
            action,
            reward,
            terminated,
            trancuated,
        });
    }

    fn build(collection_bound: CollectionBound) -> Self;
}

pub struct FixedSizeStateBuffer<E: Env> {
    pub capacity: usize,
    pub states: AllocRingBuffer<E::Tensor>,
    pub next_states: AllocRingBuffer<E::Tensor>,
    pub action: AllocRingBuffer<E::Tensor>,
    pub rewards: AllocRingBuffer<f32>,
    pub terminated: AllocRingBuffer<bool>,
    pub trancuated: AllocRingBuffer<bool>,
}

impl<E: Env> FixedSizeStateBuffer<E> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            states: AllocRingBuffer::new(capacity),
            next_states: AllocRingBuffer::new(capacity),
            rewards: AllocRingBuffer::new(capacity),
            action: AllocRingBuffer::new(capacity),
            terminated: AllocRingBuffer::new(capacity),
            trancuated: AllocRingBuffer::new(capacity),
        }
    }

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

    fn total_steps(&self) -> usize {
        self.states.len()
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
        let capacity = collection_bound.min_steps();
        Self::new(capacity)
    }

    fn last_state(&self) -> Option<Self::Tensor> {
        self.next_states.back().cloned()
    }
}

#[derive(Clone)]
pub struct VariableSizedStateBuffer<T: R2lTensor> {
    pub states: Vec<T>,
    pub next_states: Vec<T>,
    pub rewards: Vec<f32>,
    pub actions: Vec<T>,
    pub terminated: Vec<bool>,
    pub trancuated: Vec<bool>,
}

impl<T: R2lTensor> Default for VariableSizedStateBuffer<T> {
    fn default() -> Self {
        Self {
            states: vec![],
            next_states: vec![],
            rewards: vec![],
            actions: vec![],
            terminated: vec![],
            trancuated: vec![],
        }
    }
}

impl<T: R2lTensor> VariableSizedStateBuffer<T> {
    pub fn push(
        &mut self,
        state: T,
        next_state: T,
        action: T,
        reward: f32,
        terminated: bool,
        trancuated: bool,
    ) {
        self.states.push(state);
        self.next_states.push(next_state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.terminated.push(terminated);
        self.trancuated.push(trancuated);
    }
}

impl<T: R2lTensor> Buffer for VariableSizedStateBuffer<T> {
    type Tensor = T;

    fn states(&self) -> Vec<Self::Tensor> {
        self.states.iter().cloned().collect()
    }

    fn total_steps(&self) -> usize {
        self.states.len()
    }

    fn next_states(&self) -> Vec<Self::Tensor> {
        self.next_states.iter().cloned().collect()
    }

    fn actions(&self) -> Vec<Self::Tensor> {
        self.actions.iter().cloned().collect()
    }

    fn rewards(&self) -> Vec<f32> {
        self.rewards.clone()
    }

    fn terminated(&self) -> Vec<bool> {
        self.terminated.clone()
    }

    fn trancuated(&self) -> Vec<bool> {
        self.trancuated.clone()
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

    fn last_state(&self) -> Option<Self::Tensor> {
        self.next_states.last().cloned()
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

#[derive(Debug, Clone)]
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

pub fn calculate_advantages_and_returns<
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

pub struct BufferStackSampler<T: R2lTensor> {
    observations: Vec<Vec<T>>,
    actions: Vec<Vec<T>>,
}

impl<T: R2lTensor> BufferStackSampler<T> {
    pub fn sample(&self, indicies: &[(usize, usize)]) -> (Vec<T>, Vec<T>) {
        let mut observations = vec![];
        let mut actions = vec![];
        for (buf_idx, idx) in indicies {
            observations.push(self.observations[*buf_idx][*idx].clone());
            actions.push(self.actions[*buf_idx][*idx].clone());
        }
        (observations, actions)
    }
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

    pub fn sampler<T: R2lTensor + From<B::Tensor>>(&self) -> BufferStackSampler<T> {
        match self {
            Self::RefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                let states = buffers
                    .iter()
                    .map(|buffers| buffers.convert_states().clone())
                    .collect();
                let actions = buffers
                    .iter()
                    .map(|buffers| buffers.convert_actions().clone())
                    .collect();
                BufferStackSampler {
                    observations: states,
                    actions,
                }
            }
            Self::AtomicRefCounted(buffers) => {
                let buffers = buffers.iter().map(|b| b.buffer()).collect::<Vec<_>>();
                let states = buffers
                    .iter()
                    .map(|buffers| buffers.convert_states().clone())
                    .collect();
                let actions = buffers
                    .iter()
                    .map(|buffers| buffers.convert_actions().clone())
                    .collect();

                BufferStackSampler {
                    observations: states,
                    actions,
                }
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
