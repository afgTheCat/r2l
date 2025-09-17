pub mod buffers;
pub mod env_pools;

use crate::{
    distributions::Policy,
    env::{Env, Memory, Sampler2, SnapShot},
    rng::RNG,
    sampler::PolicyWrapper,
    sampler2::env_pools::builder::{BufferKind, EnvPoolType},
};
use anyhow::Result;
use rand::Rng;
use std::{fmt::Debug, marker::PhantomData};

pub trait Buffer: Sized {
    type Tensor: Clone + Send + Sync + Debug + 'static;

    // TODO: depracate this
    fn all_states(&self) -> &[Self::Tensor] {
        todo!()
    }

    fn all_states_with_converter<'a, T: 'a>(
        &'a self,
        converter: impl Fn(&Self::Tensor) -> &T,
    ) -> impl Iterator<Item = &'a T> {
        self.all_states().iter().map(move |c| converter(c))
    }

    fn rewards(&self) -> &[f32];

    fn dones(&self) -> &[bool] {
        todo!()
    }

    fn total_steps(&self) -> usize {
        todo!()
    }

    fn actions(&self) -> &[Self::Tensor] {
        todo!()
    }

    fn last_state(&self) -> Option<Self::Tensor>;

    fn push(&mut self, snapshot: Memory<Self::Tensor>);

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
}

pub struct BufferConverter<'a, B: Buffer, T: Clone + Send + Sync + Debug + 'static> {
    buffer: &'a B,
    _tensor: PhantomData<T>,
}

impl<'a, B: Buffer, T: Clone + Send + Sync + Debug + 'static> BufferConverter<'a, B, T> {
    pub fn new(buffer: &'a B) -> Self
    where
        <B as Buffer>::Tensor: Into<T>,
    {
        Self {
            buffer,
            _tensor: PhantomData,
        }
    }
}

impl<'a, B: Buffer, T: Clone + Send + Sync + Debug + 'static> Buffer for BufferConverter<'a, B, T>
where
    <B as Buffer>::Tensor: Into<T>,
{
    type Tensor = T;

    fn all_states(&self) -> &[Self::Tensor] {
        todo!()
    }

    fn rewards(&self) -> &[f32] {
        todo!()
    }

    fn dones(&self) -> &[bool] {
        todo!()
    }

    fn last_state(&self) -> Option<Self::Tensor> {
        todo!()
    }

    fn push(&mut self, snapshot: Memory<Self::Tensor>) {
        todo!()
    }

    fn last_state_terminates(&self) -> bool {
        todo!()
    }
}

pub trait Preprocessor<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> {
    fn preprocess_states(&mut self, policy: &dyn Policy<Tensor = E::Tensor>, buffers: &mut Vec<B>) {
    }
}

pub struct EmptyPreProcessor;

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> Preprocessor<E, B> for EmptyPreProcessor {}

// TODO: we need better names for this. StepBound is basically step n times, while episode bound
// basically says step steps times at least until the last state is done
#[derive(Debug, Clone)]
pub enum CollectionBound {
    StepBound { steps: usize },
    EpisodeBound { steps: usize },
}

pub struct R2lSampler2<E: Env> {
    env_pool: EnvPoolType<E>,
    preprocessor: Option<Box<dyn Preprocessor<E, BufferKind<E>>>>,
}

impl<E: Env> R2lSampler2<E> {
    pub fn new(
        env_pool: EnvPoolType<E>,
        preprocessor: Option<Box<dyn Preprocessor<E, BufferKind<E>>>>,
    ) -> Self {
        Self {
            env_pool,
            preprocessor,
        }
    }
}

impl<E: Env> Sampler2 for R2lSampler2<E> {
    type E = E;
    type Buffer = BufferKind<E>;

    fn collect_rollouts<P: Policy + Clone>(&mut self, policy: P) -> Result<Vec<BufferKind<E>>>
    where
        <E as Env>::Tensor: From<P::Tensor>,
        <E as Env>::Tensor: Into<P::Tensor>,
    {
        let policy = PolicyWrapper::new(policy);
        let collection_bound = self.env_pool.collection_bound();
        self.env_pool.set_policy(policy.clone());
        if let Some(pre_processor) = &mut self.preprocessor {
            let mut current_step = 0;
            let CollectionBound::StepBound { steps } = collection_bound else {
                panic!("pre processors currently only support rollout bounds");
            };
            while current_step < steps {
                let mut buffers = self.env_pool.get_buffers();
                pre_processor.preprocess_states(&policy, &mut buffers);
                self.env_pool.single_step();
                current_step += 1;
            }
            Ok(self.env_pool.get_buffers())
        } else {
            Ok(self.env_pool.collect())
        }
    }
}
