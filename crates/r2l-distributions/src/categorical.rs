use itertools::Itertools;
use r2l_core::{
    error::{Error, Result},
    models::{Actor, ToSafetensors},
    rng::with_rng,
    tensor::R2lTensor,
};
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

use crate::networks::Network;

#[derive(Debug, Clone)]
struct Categorical<N: Network> {
    logits: N,
}

// impl<B: Backend, N: Network + Module<B>> Module<B> for Categorical<N> {
//     type Record = N::Record;
//
//     fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
//         self.logits.collect_devices(devices)
//     }
//
//     fn fork(self, device: &B::Device) -> Self {
//         Self {
//             logits: self.logits.fork(device),
//         }
//     }
//
//     fn to_device(self, device: &B::Device) -> Self {
//         Self {
//             logits: self.logits.to_device(device),
//         }
//     }
//
//     fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
//         self.logits.visit(visitor);
//     }
//
//     fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
//         Self {
//             logits: self.logits.map(mapper),
//         }
//     }
//
//     fn load_record(self, record: Self::Record) -> Self {
//         Self {
//             logits: self.logits.load_record(record),
//         }
//     }
//
//     fn into_record(self) -> Self::Record {
//         self.logits.into_record()
//     }
// }

impl<N: Network + ToSafetensors> ToSafetensors for Categorical<N> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        self.logits.to_safetensors()
    }
}

impl<T: R2lTensor, N: Network<Tensor = T>> Actor for Categorical<N> {
    type Tensor = T;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observation);
        let action_probs = logits.sotfmax()?.to_vec()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = with_rng(|rng| distribution.sample(rng));
        Ok(T::from_vec_and_shape(vec![action as f32], vec![1])?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observation);
        let logits = logits.to_vec()?;
        // TODO: ensure that the size of the logits can not be 0
        let action = logits
            .iter()
            .position_max_by(|a, b| a.total_cmp(b))
            .unwrap();
        Ok(T::from_vec_and_shape(vec![action as f32], vec![1])?)
    }
}
