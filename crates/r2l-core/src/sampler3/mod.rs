pub mod buffer_stack;
pub mod buffers;
pub mod coordinator;
pub mod preprocessor;

use crate::{
    distributions::Policy,
    env::{Env, EnvironmentDescription, Sampler3, Sampler4},
    env_builder::{EnvBuilderTrait, EnvBuilderType},
    sampler::PolicyWrapper,
    sampler3::{
        buffer_stack::BufferStack3,
        buffers::{Buffer, BufferStack, FixedSizeStateBuffer},
        coordinator::{CoordinatorS, Location},
    },
};

// TODO: we need better names for this. StepBound is basically step n times, while episode bound
// basically says step steps times at least until the last state is done
#[derive(Debug, Clone)]
pub enum CollectionBound {
    StepBound { steps: usize },
    EpisodeBound { steps: usize },
}

impl CollectionBound {
    pub fn min_steps(&self) -> usize {
        match self {
            Self::StepBound { steps } => *steps,
            Self::EpisodeBound { steps } => *steps,
        }
    }
}

pub trait PreprocessorX<B: Buffer> {
    fn preprocess_states(&mut self, buffers: BufferStack<B>);
}

// How do I build this shit?
pub struct R2lSamplerX<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> = FixedSizeStateBuffer<E>> {
    // something like this, maybe we
    preprocessor: Option<Box<dyn PreprocessorX<B>>>,
    coordinator: CoordinatorS<E, B>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Send + 'static> R2lSamplerX<E, B> {
    pub fn build_arc<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        collection_bound: CollectionBound,
        preprocessor: Option<Box<dyn PreprocessorX<B>>>,
    ) -> Self {
        Self {
            coordinator: CoordinatorS::build_arc(env_builder, collection_bound),
            preprocessor,
        }
    }

    pub fn build_rc<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        collection_bound: CollectionBound,
        preprocessor: Option<Box<dyn PreprocessorX<B>>>,
    ) -> Self {
        Self {
            coordinator: CoordinatorS::build_rc(env_builder, collection_bound),
            preprocessor,
        }
    }

    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType<EB>,
        collection_bound: CollectionBound,
        preprocessor: Option<Box<dyn PreprocessorX<B>>>,
        location: Location,
    ) -> Self {
        match location {
            Location::Vec => Self::build_rc(env_builder, collection_bound, preprocessor),
            Location::Thread => Self::build_arc(env_builder, collection_bound, preprocessor),
        }
    }

    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        self.coordinator.env_description()
    }
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> Sampler3 for R2lSamplerX<E, B> {
    type E = E;
    type Buffer = B;

    fn collect_rollouts<P: Policy + Clone>(&mut self, policy: P) -> BufferStack<Self::Buffer>
    where
        <Self::Buffer as Buffer>::Tensor: From<P::Tensor>,
        <Self::Buffer as Buffer>::Tensor: Into<P::Tensor>,
    {
        let policy = PolicyWrapper::new(policy);
        self.coordinator.set_policy(policy);
        let collection_bound = self.coordinator.collection_bound();
        if let Some(pre_processor) = &mut self.preprocessor {
            let mut current_step = 0;
            let CollectionBound::StepBound { steps } = collection_bound else {
                panic!("pre processors currently only support rollout bounds");
            };
            while current_step < steps {
                // TODO: we need a better mechanism for pre procesors
                let buffers = self.coordinator.get_buffers();
                pre_processor.preprocess_states(buffers);
                self.coordinator.single_step();
                current_step += 1;
            }
            self.coordinator.get_buffers()
        } else {
            self.coordinator.collect();
            self.coordinator.get_buffers()
        }
    }
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor>> Sampler4 for R2lSamplerX<E, B> {
    type Env = E;

    fn collect_rollouts<P: Policy<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        policy: P,
    ) {
        let policy = PolicyWrapper::new(policy);
        self.coordinator.set_policy(policy);
        let collection_bound = self.coordinator.collection_bound();
        if let Some(pre_processor) = &mut self.preprocessor {
            let mut current_step = 0;
            let CollectionBound::StepBound { steps } = collection_bound else {
                panic!("pre processors currently only support rollout bounds");
            };
            while current_step < steps {
                // TODO: we need a better mechanism for pre procesors
                let buffers = self.coordinator.get_buffers();
                pre_processor.preprocess_states(buffers);
                self.coordinator.single_step();
                current_step += 1;
            }
        } else {
            self.coordinator.collect();
        }
    }

    fn get_buffer_stack<T: crate::tensor::R2lTensor + From<<Self::Env as Env>::Tensor>>(
        &self,
    ) -> BufferStack3<T> {
        self.coordinator.get_buffers2()
    }
}
