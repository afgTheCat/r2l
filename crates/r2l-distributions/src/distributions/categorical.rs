use itertools::Itertools;
use r2l_core::{
    error::{Error, Result},
    models::{Actor, ToSafetensors},
    rng::with_rng,
    tensor::R2lTensor,
};
use rand_distr::Distribution;
use rand_distr::weighted::WeightedIndex;

use crate::{Policy2, networks::Network};

/// Categorical policy over a network producing a nonempty vector of category logits.
///
/// Action methods accept `[1, features]`; learning methods accept `[batch, features]`.
/// Networks must honor their declared output shape; violating that contract may panic.
#[derive(Debug, Clone)]
pub struct Categorical<N: Network> {
    logits: N,
}

impl<N: Network> Categorical<N> {
    /// Builds a categorical policy after validating the network's shape contract.
    ///
    /// # Arguments
    ///
    /// * `logits` - Network whose per-observation output shape is `[categories]`.
    ///
    /// # Errors
    ///
    /// Returns an error for outputs other than a one-dimensional vector with
    /// at least one category.
    pub fn new(logits: N) -> Result<Self> {
        let output = logits.output_shape();
        if output.rank() == 1 && output.num_elements() > 0 {
            Ok(Self { logits })
        } else {
            Err(Error::invalid_parameter(
                "categorical output shape",
                "[positive category count]",
                format!("{output:?}"),
            ))
        }
    }

    fn single_logits(&self, observation: N::Tensor) -> Result<N::Tensor> {
        super::single_observation(&observation)?;
        self.logits.forward(observation)
    }
}

impl<N: Network + ToSafetensors> ToSafetensors for Categorical<N> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        self.logits.to_safetensors()
    }
}

impl<T: R2lTensor, N: Network<Tensor = T>> Actor for Categorical<N> {
    type Tensor = T;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.single_logits(observation)?;
        let action_probs = logits.softmax(1)?.to_vec()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = with_rng(|rng| distribution.sample(rng));
        Ok(T::from_vec_like(vec![action as f32], [1, 1], &logits)?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.single_logits(observation)?;
        let values = logits.to_vec()?;
        let action = values
            .iter()
            .position_max_by(|a, b| a.total_cmp(b))
            .expect("categorical network must honor its nonempty output shape");
        Ok(T::from_vec_like(vec![action as f32], [1, 1], &logits)?)
    }
}

impl<T: R2lTensor, N: Network<Tensor = T>> Policy2 for Categorical<N> {
    fn action_shape(&self) -> r2l_core::Shape {
        [1].into()
    }

    /// Evaluates `[batch, categories]` logits and `[batch, 1]` actions,
    /// returning log probabilities with shape `[batch, 1]`.
    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations)?;
        super::action_batch(&actions, logits.to_shape()[0], 1)?;
        let log_probs = logits.log_softmax(1)?;
        Ok(log_probs.gather(1, &actions)?)
    }

    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations)?;
        let probs = logits.softmax(1)?;
        let log_probs = logits.log_softmax(1)?;
        Ok(probs.mul(&log_probs)?.neg()?.sum_dim(1)?.mean()?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
