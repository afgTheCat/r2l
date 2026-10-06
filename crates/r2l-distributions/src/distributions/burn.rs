//! Burn parameter traversal and inference conversion for the generic distributions.

use burn::{
    module::{
        Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper, ModuleVisitor, Param,
    },
    tensor::{Device, Tensor},
};
use r2l_core::tensor::R2lTensor;

use super::{
    TensorParameter, bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical, policy::DistributionKind,
    sde::StateDependentNoise,
};
use crate::Network;

fn actor_bytes<M: Module>(module: &M) -> r2l_core::error::Result<Vec<u8>> {
    use burn_store::{ModuleStore, SafetensorsStore};
    use r2l_core::error::Error;
    let mut store = SafetensorsStore::default();
    store.collect_from(module).map_err(Error::wrap)?;
    store.get_bytes().map_err(Error::wrap)
}

macro_rules! actor_artifact {
    ($ty:ty) => {
        impl<N: Network<Tensor = Tensor<2>> + Module> r2l_core::models::ToSafetensors for $ty {
            fn to_safetensors(&self) -> r2l_core::error::Result<Vec<u8>> {
                actor_bytes::<_>(self)
            }
        }
    };
}

actor_artifact!(Categorical<N>);
actor_artifact!(MultiBernoulli<N>);
actor_artifact!(MultiCategorical<N>);
actor_artifact!(DistributionKind<N, Param<Tensor<2>>>);
actor_artifact!(DiagGaussian<N, Param<Tensor<2>>>);
actor_artifact!(Composite<N, Param<Tensor<2>>>);

impl<N: Network<Tensor = Tensor<2>> + Module> DistributionKind<N, Param<Tensor<2>>> {
    /// Loads actor parameters from a matching safetensors artifact.
    ///
    /// # Errors
    /// Returns an error if parameters are missing, incompatible or undecodable.
    pub fn load_from_bytes(mut self, bytes: Vec<u8>) -> r2l_core::error::Result<Self> {
        use burn_store::{ModuleSnapshot, SafetensorsStore};
        self.load_from(&mut SafetensorsStore::from_bytes(Some(bytes)))
            .map_err(r2l_core::error::Error::wrap)?;
        Ok(self)
    }
}

macro_rules! network_artifact {
    ($ty:ty) => {
        impl r2l_core::models::ToSafetensors for $ty {
            fn to_safetensors(&self) -> r2l_core::error::Result<Vec<u8>> {
                actor_bytes::<_>(self)
            }
        }
    };
}

network_artifact!(crate::networks::burn::mlp::Mlp);
network_artifact!(crate::networks::burn::cnn::Cnn);
network_artifact!(crate::networks::burn::NetworkKind);

impl TensorParameter<Tensor<2>> for Param<Tensor<2>> {
    fn value(&self) -> Tensor<2> {
        self.val()
    }

    fn for_inference(&self) -> Self {
        self.valid()
    }
}

// These policies contain one trainable network and optional fixed action metadata.
macro_rules! network_module {
    ($policy:ident $(, $metadata:ident)*) => {
        impl<N: Network<Tensor = Tensor<2>> + Module> Module for $policy<N> {
            fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
                self.logits.collect_devices(devices)
            }

            fn fork(self, device: &Device) -> Self {
                Self { logits: self.logits.fork(device), $( $metadata: self.$metadata, )* }
            }

            fn to_device(self, device: &Device) -> Self {
                Self { logits: self.logits.to_device(device), $( $metadata: self.$metadata, )* }
            }

            fn train(self) -> Self {
                Self { logits: self.logits.train(), $( $metadata: self.$metadata, )* }
            }

            fn valid(&self) -> Self {
                Self { logits: self.logits.valid(), $( $metadata: self.$metadata.clone(), )* }
            }

            fn materialize(self) -> Self {
                Self { logits: self.logits.materialize(), $( $metadata: self.$metadata, )* }
            }

            fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
                visitor.enter_module("logits", concat!("Struct:", stringify!($policy)));
                self.logits.visit(visitor);
                visitor.exit_module("logits", concat!("Struct:", stringify!($policy)));
            }

            fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
                mapper.enter_module("logits", concat!("Struct:", stringify!($policy)));
                let logits = self.logits.map(mapper);
                mapper.exit_module("logits", concat!("Struct:", stringify!($policy)));
                Self { logits, $( $metadata: self.$metadata, )* }
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

impl<N: Network<Tensor = Tensor<2>> + Module> Module for DiagGaussian<N, Param<Tensor<2>>> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.log_std
            .collect_devices(self.mean.collect_devices(devices))
    }

    fn fork(self, device: &Device) -> Self {
        Self {
            mean: self.mean.fork(device),
            log_std: self.log_std.fork(device),
            sde: self.sde.map(|sde| StateDependentNoise::new(sde.config)),
        }
    }

    fn to_device(self, device: &Device) -> Self {
        Self {
            mean: self.mean.to_device(device),
            log_std: self.log_std.to_device(device),
            sde: self.sde.map(|sde| StateDependentNoise::new(sde.config)),
        }
    }

    fn train(self) -> Self {
        Self {
            mean: self.mean.train(),
            log_std: self.log_std.train(),
            sde: self.sde.map(|sde| StateDependentNoise::new(sde.config)),
        }
    }

    fn valid(&self) -> Self {
        Self {
            mean: self.mean.valid(),
            log_std: self.log_std.valid(),
            sde: self
                .sde
                .as_ref()
                .map(|sde| StateDependentNoise::new(sde.config)),
        }
    }

    fn materialize(self) -> Self {
        Self {
            mean: self.mean.materialize(),
            log_std: self.log_std.materialize(),
            ..self
        }
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.enter_module("mean", "Struct:DiagGaussian");
        self.mean.visit(visitor);
        visitor.exit_module("mean", "Struct:DiagGaussian");
        visitor.enter_module("log_std", "Struct:DiagGaussian");
        self.log_std.visit(visitor);
        visitor.exit_module("log_std", "Struct:DiagGaussian");
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.enter_module("mean", "Struct:DiagGaussian");
        let mean = self.mean.map(mapper);
        mapper.exit_module("mean", "Struct:DiagGaussian");
        mapper.enter_module("log_std", "Struct:DiagGaussian");
        let log_std = Module::map(self.log_std, mapper);
        mapper.exit_module("log_std", "Struct:DiagGaussian");
        Self {
            mean,
            log_std,
            sde: self.sde.map(|sde| StateDependentNoise::new(sde.config)),
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

impl<N: Network<Tensor = Tensor<2>> + Module> Module for DistributionKind<N, Param<Tensor<2>>> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        dispatch!(self, policy => policy.collect_devices(devices))
    }

    fn fork(self, device: &Device) -> Self {
        map_distribution!(self, fork(device))
    }

    fn to_device(self, device: &Device) -> Self {
        map_distribution!(self, to_device(device))
    }

    fn train(self) -> Self {
        map_distribution!(self, train())
    }

    fn valid(&self) -> Self {
        map_distribution!(self, valid())
    }

    fn materialize(self) -> Self {
        map_distribution!(self, materialize())
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        dispatch!(self, policy => policy.visit(visitor));
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        map_distribution!(self, map(mapper))
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

impl<N: Network<Tensor = Tensor<2>> + Module> Module for Composite<N, Param<Tensor<2>>> {
    fn collect_devices(&self, devices: Vec<Device>) -> Vec<Device> {
        self.policies.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        Self {
            policies: self.policies.fork(device),
            ..self
        }
    }

    fn to_device(self, device: &Device) -> Self {
        Self {
            policies: self.policies.to_device(device),
            ..self
        }
    }

    fn train(self) -> Self {
        Self {
            policies: self.policies.train(),
            ..self
        }
    }

    fn valid(&self) -> Self {
        Self {
            policies: self.policies.valid(),
            action_sizes: self.action_sizes.clone(),
            action_size: self.action_size,
        }
    }

    fn materialize(self) -> Self {
        Self {
            policies: self.policies.materialize(),
            ..self
        }
    }

    fn visit<V: ModuleVisitor>(&self, visitor: &mut V) {
        visitor.enter_module("policies", "Struct:Composite");
        self.policies.visit(visitor);
        visitor.exit_module("policies", "Struct:Composite");
    }

    fn map<M: ModuleMapper>(self, mapper: &mut M) -> Self {
        mapper.enter_module("policies", "Struct:Composite");
        let policies = self.policies.map(mapper);
        mapper.exit_module("policies", "Struct:Composite");
        Self { policies, ..self }
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
