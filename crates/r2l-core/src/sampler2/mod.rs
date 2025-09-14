pub mod buffers;
pub mod env_pools;
// pub mod holders;

use crate::{
    distributions::Policy,
    env::{Env, Memory, SnapShot},
    rng::RNG,
    sampler::PolicyWrapper,
};
use rand::Rng;
use std::fmt::Debug;

// This version V3.
pub trait Buffer {
    type E: Env;

    fn last_state(&self) -> Option<<Self::E as Env>::Tensor>;

    fn push(&mut self, snapshot: Memory<<Self::E as Env>::Tensor>);

    fn last_state_terminates(&self) -> bool;

    #[inline(always)]
    fn step(
        &mut self,
        env: &mut Self::E,
        distr: &Box<dyn Policy<Tensor = <Self::E as Env>::Tensor>>,
        last_state: Option<<Self::E as Env>::Tensor>,
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

trait EnvPool {
    type E: Env;
    type B: Buffer<E = Self::E>;

    fn collection_bound(&self) -> CollectionBound;

    fn set_policy<P: Policy<Tensor = <Self::E as Env>::Tensor> + Clone>(&mut self, policy: P);

    fn get_buffers(&self) -> Vec<Self::B>;

    fn single_step(&mut self);

    fn collect(&self) -> Vec<Self::B>;
}

pub trait Preprocessor<E: Env, B: Buffer<E = E>> {
    fn preprocess_states(&mut self, policy: &dyn Policy<Tensor = E::Tensor>, buffers: &mut Vec<B>) {
    }
}

pub struct EmptyPreProcessor;

impl<E: Env, B: Buffer<E = E>> Preprocessor<E, B> for EmptyPreProcessor {}

// TODO: we need better names for this. StepBound is basically step n times, while episode bound
// basically says step steps times at least until the last state is done
#[derive(Debug, Clone)]
pub enum CollectionBound {
    StepBound { steps: usize },
    EpisodeBound { steps: usize },
}

pub struct R2lSampler2<EP: EnvPool, P: Preprocessor<EP::E, EP::B>> {
    env_pool: EP,
    preprocessor: Option<P>,
}

impl<EP: EnvPool, PR: Preprocessor<EP::E, EP::B>> R2lSampler2<EP, PR> {
    fn collect_rollouts<P: Policy + Clone>(&mut self, policy: P) -> Vec<<EP as EnvPool>::B>
    where
        <<EP as EnvPool>::E as Env>::Tensor: From<P::Tensor>,
        <<EP as EnvPool>::E as Env>::Tensor: Into<P::Tensor>,
    {
        let policy: PolicyWrapper<P, <<EP as EnvPool>::E as Env>::Tensor> =
            PolicyWrapper::new(policy);
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
            self.env_pool.get_buffers()
        } else {
            self.env_pool.collect()
        }
    }
}
