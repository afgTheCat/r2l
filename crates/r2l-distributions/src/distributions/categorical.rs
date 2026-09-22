use itertools::Itertools;
use r2l_core::{
    error::{Error, Result, TensorError},
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

impl<N: Network + ToSafetensors> ToSafetensors for Categorical<N> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        self.logits.to_safetensors()
    }
}

impl<T: R2lTensor, N: Network<Tensor = T>> Actor for Categorical<N> {
    type Tensor = T;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observation);
        let shape = logits.to_shape();
        let dim = shape.len().saturating_sub(1);
        let action_probs = logits.softmax(dim)?.to_vec()?;
        let distribution = WeightedIndex::new(&action_probs).map_err(Error::wrap)?;
        let action = with_rng(|rng| distribution.sample(rng));
        Ok(T::from_vec_and_shape(
            vec![action as f32],
            vec![1; shape.len()],
        )?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observation);
        let shape = logits.to_shape();
        let logits = logits.to_vec()?;
        let action = logits
            .iter()
            .position_max_by(|a, b| a.total_cmp(b))
            .ok_or_else(|| TensorError::EmptyInput {
                operation: "select categorical modal action".into(),
            })?;
        Ok(T::from_vec_and_shape(
            vec![action as f32],
            vec![1; shape.len()],
        )?)
    }
}

/// Trainable action distribution interface used by on-policy algorithms.
///
/// A `Policy` extends [`Actor`] with the quantities needed to compute policy
/// gradient losses and entropy bonuses over a batch.
pub trait Policy2: Actor {
    /// Computes log probabilities for batched observation/action pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the policy cannot evaluate the batch.
    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor>;

    /// Returns a representative action standard deviation when available.
    ///
    /// # Errors
    ///
    /// Returns an error if the standard deviation cannot be computed.
    /// Returns `Ok(None)` when the policy has no meaningful scalar standard deviation.
    fn std(&self) -> Result<Option<f32>>;

    /// Computes the policy entropy for a batch of states.
    ///
    /// # Errors
    ///
    /// Returns an error if the entropy cannot be computed.
    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor>;
}

impl<T: R2lTensor, N: Network<Tensor = T>> Policy2 for Categorical<N> {
    /// Evaluates `[batch, categories]` logits and `[batch, 1]` actions,
    /// returning log probabilities with shape `[batch, 1]`.
    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations);
        let log_probs = logits.log_softmax(1)?;
        Ok(log_probs.gather(1, &actions)?)
    }

    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations);
        let probs = logits.softmax(1)?;
        let log_probs = logits.log_softmax(1)?;
        // Scaling the elementwise mean sums over categories and averages over states.
        let categories = logits.to_shape()[1];
        Ok(probs
            .mul(&log_probs)?
            .neg()?
            .mean()?
            .mul_scalar(categories as f32)?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
