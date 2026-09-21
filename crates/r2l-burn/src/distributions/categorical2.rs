use std::marker::PhantomData;

use burn::module::ModuleDisplay;
use burn::tensor::TensorData;
use burn::tensor::activation::{log_softmax, softmax};
use burn::{Tensor, module::Module, tensor::backend::Backend};
use burn_store::{ModuleStore, SafetensorsStore};
use itertools::Itertools;
use r2l_core::error::{Error, InvalidParameterError, Result, TensorError};
use r2l_core::models::{ActivationFunction, Actor, Policy, ToSafetensors};
use r2l_core::rng::with_rng;
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

use crate::networks::Network;
use crate::networks::mlp::Mlp;

// TODO: we probably want Network here, not Module. Whatever
#[derive(Debug, Module)]
pub struct CategoricalDistribution2<B: Backend, N: Module<B>> {
    logits: N,
    _b: PhantomData<B>,
}

impl<B: Backend> CategoricalDistribution2<B, Mlp<B>> {
    pub fn build_mlp(logits_layers: &[usize], activation: ActivationFunction) -> Result<Self> {
        if logits_layers.is_empty() {
            return Err(Error::InvalidParameter(Box::new(
                InvalidParameterError::InvalidValue {
                    name: "logits_layers".into(),
                    expected: "at least one layer".into(),
                    value: "[]".into(),
                },
            )));
        }
        let logits: Mlp<B> = Mlp::build(logits_layers, activation);
        Ok(Self {
            logits,
            _b: PhantomData,
        })
    }
}

impl<B: Backend, N: Module<B> + ModuleDisplay> ToSafetensors for CategoricalDistribution2<B, N> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        let mut store = SafetensorsStore::default();
        store.collect_from(self).map_err(Error::wrap)?;
        store.get_bytes().map_err(Error::wrap)
    }
}

impl<B: Backend, N: Network<B>> Actor for CategoricalDistribution2<B, N> {
    type Tensor = Tensor<B, 1>;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observation);
        let action_probs: Vec<f32> = softmax(logits, 1)
            .to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("read categorical probabilities", error))?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = with_rng(|rng| distribution.sample(rng));
        let action = Tensor::from_data(
            TensorData::new(vec![action as f32], vec![1]),
            &Default::default(),
        );
        Ok(action)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits: Vec<f32> = self
            .logits
            .forward(observation)
            .to_data()
            .to_vec()
            .map_err(|error| TensorError::operation("read categorical logits", error))?;
        let action = logits
            .iter()
            .position_max_by(|a, b| a.total_cmp(b))
            .ok_or_else(|| TensorError::EmptyInput {
                operation: "select categorical modal action".into(),
            })?;
        let action = Tensor::from_data(
            TensorData::new(vec![action as f32], vec![1]),
            &Default::default(),
        );
        Ok(action)
    }
}

impl<B: Backend, N: Network<B>> Policy for CategoricalDistribution2<B, N> {
    fn log_probs(
        &self,
        observations: &[Self::Tensor],
        actions: &[Self::Tensor],
    ) -> Result<Self::Tensor> {
        debug_assert!(!observations.is_empty());
        debug_assert_eq!(observations.len(), actions.len());
        let logits = self.logits.batch_forward(observations);
        let actions: Tensor<B, 2> = Tensor::stack(actions.to_vec(), 0);
        let log_probs = log_softmax(logits, 1);
        Ok(log_probs.gather(1, actions.int()).squeeze_dim::<1>(1))
    }

    fn entropy(&self, observations: &[Self::Tensor]) -> Result<Self::Tensor> {
        let logits = self.logits.batch_forward(observations);
        let probs = softmax(logits.clone(), 1);
        let log_probs = log_softmax(logits, 1);
        let entropy_per_state = (probs * log_probs).neg().sum_dim(1);
        let entropy = entropy_per_state.mean();
        Ok(entropy)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
