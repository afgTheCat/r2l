mod buffers;
mod coordinator;

use crate::{
    distributions::Policy,
    env::{Env, EnvBuilderTrait},
    sampler::PolicyWrapper,
    sampler2::{Buffer, CollectionBound, env_pools::builder::EnvBuilderType2},
    sampler3::{
        buffers::BufferStack,
        coordinator::{CoordinatorS, Location},
    },
};

pub trait PreprocessorX<B: Buffer> {
    fn preprocess_states(&mut self, buffers: BufferStack<B>);
}

// How do I build this shit?
struct R2lSamplerX<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone> {
    // something like this, maybe we
    preprocessor: Option<Box<dyn PreprocessorX<B>>>,
    coordinator: CoordinatorS<E, B>,
}

impl<E: Env, B: Buffer<Tensor = <E as Env>::Tensor> + Clone + Send + 'static> R2lSamplerX<E, B> {
    pub fn build_arc<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
        preprocessor: Option<Box<dyn PreprocessorX<B>>>,
    ) -> Self {
        Self {
            coordinator: CoordinatorS::build_arc(env_builder, collection_bound),
            preprocessor,
        }
    }

    pub fn build_rc<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
        preprocessor: Option<Box<dyn PreprocessorX<B>>>,
    ) -> Self {
        Self {
            coordinator: CoordinatorS::build_rc(env_builder, collection_bound),
            preprocessor,
        }
    }

    pub fn build<EB: EnvBuilderTrait<Env = E>>(
        env_builder: EnvBuilderType2<EB>,
        collection_bound: CollectionBound,
        preprocessor: Option<Box<dyn PreprocessorX<B>>>,
        location: Location,
    ) -> Self {
        match location {
            Location::Vec => Self::build_rc(env_builder, collection_bound, preprocessor),
            Location::Thread => Self::build_arc(env_builder, collection_bound, preprocessor),
        }
    }

    fn collect<P: Policy + Clone>(&mut self, policy: P) -> BufferStack<B>
    where
        <E as Env>::Tensor: From<P::Tensor>,
        <E as Env>::Tensor: Into<P::Tensor>,
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
