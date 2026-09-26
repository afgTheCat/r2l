//! Candle actor artifacts use deterministic names matching the space/network builders.

use candle_core::Tensor;
use r2l_core::{
    error::{Error, Result},
    models::ToSafetensors,
};

use super::{
    bernoulli::MultiBernoulli, categorical::Categorical, composite::Composite,
    diagonal::DiagGaussian, multi_categorical::MultiCategorical, policy::DistributionKind,
};
use crate::networks::candle_mlp::Mlp;

impl DistributionKind<Mlp> {
    fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            Self::Categorical(policy) => policy.logits.named_tensors(prefix),
            Self::MultiBernoulli(policy) => policy.logits.named_tensors(prefix),
            Self::MultiCategorical(policy) => policy.logits.named_tensors(prefix),
            Self::DiagGaussian(policy) => policy.named_tensors(prefix),
            Self::Composite(policy) => policy.named_tensors(prefix),
        }
    }
}

impl DiagGaussian<Mlp> {
    fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut tensors = self.mean.named_tensors(prefix);
        tensors.push((format!("{prefix}.log_std"), self.log_std.clone()));
        tensors
    }
}

impl Composite<Mlp> {
    fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.policies
            .iter()
            .enumerate()
            .flat_map(|(index, policy)| policy.named_tensors(&format!("{prefix}.{index}")))
            .collect()
    }
}

macro_rules! network_tensors {
    ($ty:ident) => {
        impl $ty<Mlp> {
            fn named_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
                self.logits.named_tensors(prefix)
            }
        }
    };
}
network_tensors!(Categorical);
network_tensors!(MultiBernoulli);
network_tensors!(MultiCategorical);

macro_rules! actor_artifact {
    ($ty:ty) => {
        impl ToSafetensors for $ty {
            fn to_safetensors(&self) -> Result<Vec<u8>> {
                safetensors::serialize(self.named_tensors("policy"), None).map_err(Error::wrap)
            }
        }
    };
}

actor_artifact!(Mlp);
actor_artifact!(DiagGaussian<Mlp>);
actor_artifact!(Composite<Mlp>);
actor_artifact!(DistributionKind<Mlp>);

actor_artifact!(Categorical<Mlp>);
actor_artifact!(MultiBernoulli<Mlp>);
actor_artifact!(MultiCategorical<Mlp>);
