use r2l_core::{
    Shape,
    env::action_ranges,
    error::{Error, Result},
    models::Actor,
    tensor::R2lTensor,
};

use super::policy::Policy;
use crate::{Network, Policy2};

/// Independent child policies evaluated on the same observations.
/// Actions are concatenated in child order; dictionary callers should use their space's key order.
#[derive(Debug, Clone)]
pub struct Composite<N: Network> {
    policies: Vec<Policy<N>>,
    action_sizes: Vec<usize>,
    action_size: usize,
}

impl<N: Network> Composite<N> {
    /// Builds a nonempty ordered collection of policies over the same observations.
    ///
    /// # Arguments
    /// * `policies` - Children in the desired action-field order; nested composites are supported.
    ///
    /// # Errors
    /// Returns an error for an empty collection or an overflowing combined action width.
    pub fn new(policies: Vec<Policy<N>>) -> Result<Self> {
        let action_sizes: Vec<_> = policies
            .iter()
            .map(|policy| policy.action_shape().num_elements())
            .collect();
        let total = action_sizes
            .iter()
            .try_fold(0_usize, |size, &width| size.checked_add(width));
        let Some(action_size) = total.filter(|&size| size > 0) else {
            return Err(Error::invalid_parameter(
                "composite actions",
                "nonempty representable action vector",
                format!("{action_sizes:?}"),
            ));
        };
        Ok(Self {
            policies,
            action_sizes,
            action_size,
        })
    }
}

impl<N: Network> Actor for Composite<N> {
    type Tensor = N::Tensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let actions = self
            .policies
            .iter()
            .map(|policy| policy.action(observation.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::Tensor::cat(&actions, 1)?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        let actions = self
            .policies
            .iter()
            .map(|policy| policy.mode_action(observation.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::Tensor::cat(&actions, 1)?)
    }
}

impl<N: Network> Policy2 for Composite<N> {
    fn action_shape(&self) -> Shape {
        [self.action_size].into()
    }

    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        let shape = observations.to_shape();
        if shape.rank() != 2 || shape[0] == 0 {
            return Err(Error::invalid_parameter(
                "observation shape",
                "[nonzero batch, features]",
                format!("{shape:?}"),
            ));
        }
        super::action_batch(&actions, shape[0], self.action_size)?;
        let mut log_probs = Vec::with_capacity(self.policies.len());
        for (policy, (offset, width)) in self.policies.iter().zip(action_ranges(&self.action_sizes))
        {
            log_probs
                .push(policy.log_probs(observations.clone(), actions.narrow(1, offset, width)?)?);
        }
        Ok(Self::Tensor::add_multiple(&log_probs)?)
    }

    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        // Every child already averages over the batch; independent entropies add.
        let entropies = self
            .policies
            .iter()
            .map(|policy| policy.entropy(observations.clone()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::Tensor::add_multiple(&entropies)?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
