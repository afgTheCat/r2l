//! Burn parameter traversal and inference conversion for the generic distributions.

use burn::{
    module::{
        AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
        ModuleVisitor, Param,
    },
    prelude::Backend,
    record::{PrecisionSettings, Record},
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_core::tensor::R2lTensor;
use serde::{Deserialize, Serialize};

use super::{
    TensorParameter, bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical, policy::DistributionKind,
};
use crate::Network;

impl<B: Backend> TensorParameter<Tensor<B, 2>> for Param<Tensor<B, 2>> {
    fn value(&self) -> Tensor<B, 2> {
        self.val()
    }
}

// These policies contain one trainable network and optional fixed action metadata.
macro_rules! network_module {
    ($policy:ident $(, $metadata:ident)*) => {
        impl<B: Backend, N: Network<Tensor = Tensor<B, 2>> + Module<B>> Module<B> for $policy<N> {
            type Record = N::Record;

            fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
                self.logits.collect_devices(devices)
            }

            fn fork(self, device: &B::Device) -> Self {
                Self { logits: self.logits.fork(device), $( $metadata: self.$metadata, )* }
            }

            fn to_device(self, device: &B::Device) -> Self {
                Self { logits: self.logits.to_device(device), $( $metadata: self.$metadata, )* }
            }

            fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
                visitor.enter_module("logits", concat!("Struct:", stringify!($policy)));
                self.logits.visit(visitor);
                visitor.exit_module("logits", concat!("Struct:", stringify!($policy)));
            }

            fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
                mapper.enter_module("logits", concat!("Struct:", stringify!($policy)));
                let logits = self.logits.map(mapper);
                mapper.exit_module("logits", concat!("Struct:", stringify!($policy)));
                Self { logits, $( $metadata: self.$metadata, )* }
            }

            fn load_record(self, record: Self::Record) -> Self {
                Self { logits: self.logits.load_record(record), $( $metadata: self.$metadata, )* }
            }

            fn into_record(self) -> Self::Record {
                self.logits.into_record()
            }
        }

        impl<B, N> AutodiffModule<B> for $policy<N>
        where
            B: AutodiffBackend,
            N: Network<Tensor = Tensor<B, 2>> + AutodiffModule<B>,
            N::InnerModule: Network<Tensor = Tensor<B::InnerBackend, 2>>,
        {
            type InnerModule = $policy<N::InnerModule>;

            fn valid(&self) -> Self::InnerModule {
                $policy { logits: self.logits.valid(), $( $metadata: self.$metadata.clone(), )* }
            }

            fn from_inner(module: Self::InnerModule) -> Self {
                Self { logits: N::from_inner(module.logits), $( $metadata: module.$metadata, )* }
            }
        }

        impl<N: Network + ModuleDisplay> ModuleDisplayDefault for $policy<N> {
            fn content(&self, content: Content) -> Option<Content> {
                content.add("logits", &self.logits).optional()
            }

            fn num_params(&self) -> usize {
                ModuleDisplayDefault::num_params(&self.logits)
            }
        }

        impl<N: Network + ModuleDisplay> ModuleDisplay for $policy<N> {}
    };
}

network_module!(Categorical);
network_module!(MultiBernoulli);
network_module!(MultiCategorical, categories);

impl<B: Backend, N: Network<Tensor = Tensor<B, 2>> + Module<B>> Module<B>
    for DiagGaussian<N, Param<Tensor<B, 2>>>
{
    type Record = (N::Record, Param<Tensor<B, 2>>);

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        self.log_std
            .collect_devices(self.mean.collect_devices(devices))
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.fork(device),
            log_std: self.log_std.fork(device),
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            mean: self.mean.to_device(device),
            log_std: self.log_std.to_device(device),
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.enter_module("mean", "Struct:DiagGaussian");
        self.mean.visit(visitor);
        visitor.exit_module("mean", "Struct:DiagGaussian");
        visitor.enter_module("log_std", "Struct:DiagGaussian");
        self.log_std.visit(visitor);
        visitor.exit_module("log_std", "Struct:DiagGaussian");
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.enter_module("mean", "Struct:DiagGaussian");
        let mean = self.mean.map(mapper);
        mapper.exit_module("mean", "Struct:DiagGaussian");
        mapper.enter_module("log_std", "Struct:DiagGaussian");
        let log_std = Module::map(self.log_std, mapper);
        mapper.exit_module("log_std", "Struct:DiagGaussian");
        Self { mean, log_std }
    }

    fn load_record(self, (mean, log_std): Self::Record) -> Self {
        Self {
            mean: self.mean.load_record(mean),
            log_std: self.log_std.load_record(log_std),
        }
    }

    fn into_record(self) -> Self::Record {
        (self.mean.into_record(), self.log_std.into_record())
    }
}

impl<B, N> AutodiffModule<B> for DiagGaussian<N, Param<Tensor<B, 2>>>
where
    B: AutodiffBackend,
    N: Network<Tensor = Tensor<B, 2>> + AutodiffModule<B>,
    N::InnerModule: Network<Tensor = Tensor<B::InnerBackend, 2>>,
{
    type InnerModule = DiagGaussian<N::InnerModule, Param<Tensor<B::InnerBackend, 2>>>;

    fn valid(&self) -> Self::InnerModule {
        DiagGaussian {
            mean: self.mean.valid(),
            log_std: self.log_std.valid(),
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Self {
            mean: N::from_inner(module.mean),
            log_std: Param::from_inner(module.log_std),
        }
    }
}

impl<N: Network + ModuleDisplay, P: TensorParameter<N::Tensor> + ModuleDisplay> ModuleDisplayDefault
    for DiagGaussian<N, P>
{
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("mean", &self.mean)
            .add("log_std", &self.log_std)
            .optional()
    }

    fn num_params(&self) -> usize {
        ModuleDisplayDefault::num_params(&self.mean) + self.log_std.value().size()
    }
}

impl<N: Network + ModuleDisplay, P: TensorParameter<N::Tensor> + ModuleDisplay> ModuleDisplay
    for DiagGaussian<N, P>
{
}

/// Parameter records preserve the variant and nesting of composite policies.
#[derive(Clone, Serialize, Deserialize)]
pub enum DistributionRecord<N, P> {
    Categorical(N),
    DiagGaussian((N, P)),
    MultiBernoulli(N),
    MultiCategorical(N),
    Composite(Vec<DistributionRecord<N, P>>),
}

impl<B: Backend, N: Record<B>, P: Record<B>> Record<B> for DistributionRecord<N, P> {
    type Item<S: PrecisionSettings> = DistributionRecord<N::Item<S>, P::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        match self {
            Self::Categorical(record) => DistributionRecord::Categorical(record.into_item::<S>()),
            Self::DiagGaussian((mean, log_std)) => {
                DistributionRecord::DiagGaussian((mean.into_item::<S>(), log_std.into_item::<S>()))
            }
            Self::MultiBernoulli(record) => {
                DistributionRecord::MultiBernoulli(record.into_item::<S>())
            }
            Self::MultiCategorical(record) => {
                DistributionRecord::MultiCategorical(record.into_item::<S>())
            }
            Self::Composite(records) => DistributionRecord::Composite(
                records.into_iter().map(Record::into_item::<S>).collect(),
            ),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        match item {
            DistributionRecord::Categorical(record) => {
                Self::Categorical(N::from_item::<S>(record, device))
            }
            DistributionRecord::DiagGaussian((mean, log_std)) => Self::DiagGaussian((
                N::from_item::<S>(mean, device),
                P::from_item::<S>(log_std, device),
            )),
            DistributionRecord::MultiBernoulli(record) => {
                Self::MultiBernoulli(N::from_item::<S>(record, device))
            }
            DistributionRecord::MultiCategorical(record) => {
                Self::MultiCategorical(N::from_item::<S>(record, device))
            }
            DistributionRecord::Composite(records) => Self::Composite(
                records
                    .into_iter()
                    .map(|record| Self::from_item::<S>(record, device))
                    .collect(),
            ),
        }
    }
}

macro_rules! dispatch {
    ($value:expr, $policy:ident => $call:expr) => {
        match $value {
            DistributionKind::Categorical($policy) => $call,
            DistributionKind::DiagGaussian($policy) => $call,
            DistributionKind::MultiBernoulli($policy) => $call,
            DistributionKind::MultiCategorical($policy) => $call,
            DistributionKind::Composite($policy) => $call,
        }
    };
}

macro_rules! map_distribution {
    ($value:expr, $method:ident($($arg:expr),*)) => {
        match $value {
            DistributionKind::Categorical(policy) => DistributionKind::Categorical(policy.$method($($arg),*)),
            DistributionKind::DiagGaussian(policy) => DistributionKind::DiagGaussian(policy.$method($($arg),*)),
            DistributionKind::MultiBernoulli(policy) => DistributionKind::MultiBernoulli(policy.$method($($arg),*)),
            DistributionKind::MultiCategorical(policy) => DistributionKind::MultiCategorical(policy.$method($($arg),*)),
            DistributionKind::Composite(policy) => DistributionKind::Composite(policy.$method($($arg),*)),
        }
    };
}

impl<B: Backend, N: Network<Tensor = Tensor<B, 2>> + Module<B>> Module<B>
    for DistributionKind<N, Param<Tensor<B, 2>>>
{
    type Record = DistributionRecord<N::Record, Param<Tensor<B, 2>>>;

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        dispatch!(self, policy => policy.collect_devices(devices))
    }

    fn fork(self, device: &B::Device) -> Self {
        map_distribution!(self, fork(device))
    }

    fn to_device(self, device: &B::Device) -> Self {
        map_distribution!(self, to_device(device))
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        dispatch!(self, policy => policy.visit(visitor));
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        map_distribution!(self, map(mapper))
    }

    fn load_record(self, record: Self::Record) -> Self {
        match (self, record) {
            (Self::Categorical(policy), DistributionRecord::Categorical(record)) => {
                Self::Categorical(policy.load_record(record))
            }
            (Self::DiagGaussian(policy), DistributionRecord::DiagGaussian(record)) => {
                Self::DiagGaussian(policy.load_record(record))
            }
            (Self::MultiBernoulli(policy), DistributionRecord::MultiBernoulli(record)) => {
                Self::MultiBernoulli(policy.load_record(record))
            }
            (Self::MultiCategorical(policy), DistributionRecord::MultiCategorical(record)) => {
                Self::MultiCategorical(policy.load_record(record))
            }
            (Self::Composite(policy), DistributionRecord::Composite(record)) => {
                Self::Composite(policy.load_record(record))
            }
            _ => panic!("distribution record must match the policy variant"),
        }
    }

    fn into_record(self) -> Self::Record {
        match self {
            Self::Categorical(policy) => DistributionRecord::Categorical(policy.into_record()),
            Self::DiagGaussian(policy) => DistributionRecord::DiagGaussian(policy.into_record()),
            Self::MultiBernoulli(policy) => {
                DistributionRecord::MultiBernoulli(policy.into_record())
            }
            Self::MultiCategorical(policy) => {
                DistributionRecord::MultiCategorical(policy.into_record())
            }
            Self::Composite(policy) => DistributionRecord::Composite(policy.into_record()),
        }
    }
}

impl<B, N> AutodiffModule<B> for DistributionKind<N, Param<Tensor<B, 2>>>
where
    B: AutodiffBackend,
    N: Network<Tensor = Tensor<B, 2>> + AutodiffModule<B>,
    N::InnerModule: Network<Tensor = Tensor<B::InnerBackend, 2>>,
{
    type InnerModule = DistributionKind<N::InnerModule, Param<Tensor<B::InnerBackend, 2>>>;

    fn valid(&self) -> Self::InnerModule {
        map_distribution!(self, valid())
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        match module {
            DistributionKind::Categorical(policy) => {
                Self::Categorical(Categorical::from_inner(policy))
            }
            DistributionKind::DiagGaussian(policy) => {
                Self::DiagGaussian(DiagGaussian::from_inner(policy))
            }
            DistributionKind::MultiBernoulli(policy) => {
                Self::MultiBernoulli(MultiBernoulli::from_inner(policy))
            }
            DistributionKind::MultiCategorical(policy) => {
                Self::MultiCategorical(MultiCategorical::from_inner(policy))
            }
            DistributionKind::Composite(policy) => Self::Composite(Composite::from_inner(policy)),
        }
    }
}

impl<N: Network + ModuleDisplay, P: TensorParameter<N::Tensor> + ModuleDisplay> ModuleDisplayDefault
    for DistributionKind<N, P>
{
    fn content(&self, content: Content) -> Option<Content> {
        dispatch!(self, policy => policy.content(content))
    }

    fn num_params(&self) -> usize {
        dispatch!(self, policy => ModuleDisplayDefault::num_params(policy))
    }
}

impl<N: Network + ModuleDisplay, P: TensorParameter<N::Tensor> + ModuleDisplay> ModuleDisplay
    for DistributionKind<N, P>
{
}

impl<B: Backend, N: Network<Tensor = Tensor<B, 2>> + Module<B>> Module<B>
    for Composite<N, Param<Tensor<B, 2>>>
{
    type Record = Vec<DistributionRecord<N::Record, Param<Tensor<B, 2>>>>;

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        self.policies.collect_devices(devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            policies: self.policies.fork(device),
            ..self
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            policies: self.policies.to_device(device),
            ..self
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.enter_module("policies", "Struct:Composite");
        self.policies.visit(visitor);
        visitor.exit_module("policies", "Struct:Composite");
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        mapper.enter_module("policies", "Struct:Composite");
        let policies = self.policies.map(mapper);
        mapper.exit_module("policies", "Struct:Composite");
        Self { policies, ..self }
    }

    fn load_record(self, record: Self::Record) -> Self {
        assert_eq!(
            self.policies.len(),
            record.len(),
            "composite record must match the policy count"
        );
        Self {
            policies: self.policies.load_record(record),
            ..self
        }
    }

    fn into_record(self) -> Self::Record {
        self.policies.into_record()
    }
}

impl<B, N> AutodiffModule<B> for Composite<N, Param<Tensor<B, 2>>>
where
    B: AutodiffBackend,
    N: Network<Tensor = Tensor<B, 2>> + AutodiffModule<B>,
    N::InnerModule: Network<Tensor = Tensor<B::InnerBackend, 2>>,
{
    type InnerModule = Composite<N::InnerModule, Param<Tensor<B::InnerBackend, 2>>>;

    fn valid(&self) -> Self::InnerModule {
        Composite {
            policies: self.policies.valid(),
            action_sizes: self.action_sizes.clone(),
            action_size: self.action_size,
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Self {
            policies: Vec::from_inner(module.policies),
            action_sizes: module.action_sizes,
            action_size: module.action_size,
        }
    }
}

impl<N: Network + ModuleDisplay, P: TensorParameter<N::Tensor> + ModuleDisplay> ModuleDisplayDefault
    for Composite<N, P>
{
    fn content(&self, content: Content) -> Option<Content> {
        content.add("policies", &self.policies).optional()
    }

    fn num_params(&self) -> usize {
        self.policies
            .iter()
            .map(ModuleDisplayDefault::num_params)
            .sum()
    }
}

impl<N: Network + ModuleDisplay, P: TensorParameter<N::Tensor> + ModuleDisplay> ModuleDisplay
    for Composite<N, P>
{
}
