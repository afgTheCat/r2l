use itertools::Itertools;
use r2l_core::{
    Shape,
    env::action_ranges,
    error::{Error, Result},
    models::{Actor, ToSafetensors},
    rng::with_rng,
    tensor::R2lTensor,
};
use rand_distr::{Distribution, weighted::WeightedIndex};

use crate::{Network, Policy2};

/// Independent categorical actions, using consecutive groups of network logits.
#[derive(Debug, Clone)]
pub struct MultiCategorical<N: Network> {
    pub(super) logits: N,
    pub(super) categories: Vec<usize>,
}

impl<N: Network> MultiCategorical<N> {
    /// Builds a policy whose output is split into positive category counts.
    ///
    /// # Arguments
    /// * `logits` - Network producing the concatenated category logits.
    /// * `categories` - Number of categories for each action component, in order.
    ///
    /// # Errors
    /// Returns an error if counts are empty, zero, overflowing, or incompatible with the output.
    pub fn new(logits: N, categories: Vec<usize>) -> Result<Self> {
        let width = super::output_width(&logits)?;
        let total = categories
            .iter()
            .try_fold(0_usize, |sum, &n| sum.checked_add(n));
        if categories.is_empty() || categories.contains(&0) || total != Some(width) {
            return Err(Error::invalid_parameter(
                "category counts",
                format!("positive counts summing to {width}"),
                format!("{categories:?}"),
            ));
        }
        Ok(Self { logits, categories })
    }
}

impl<N: Network> Actor for MultiCategorical<N> {
    type Tensor = N::Tensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        let logits = self.logits.forward(observation)?;
        let mut actions = Vec::with_capacity(self.categories.len());
        for (offset, count) in action_ranges(&self.categories) {
            let probs = logits.narrow(1, offset, count)?.softmax(1)?.to_vec()?;
            let distribution = WeightedIndex::new(&probs).map_err(Error::wrap)?;
            actions.push(with_rng(|rng| distribution.sample(rng)) as f32);
        }
        Ok(Self::Tensor::from_vec_like(
            actions,
            [1, self.categories.len()],
            &logits,
        )?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        let logits = self.logits.forward(observation)?;
        let mut actions = Vec::with_capacity(self.categories.len());
        for (offset, count) in action_ranges(&self.categories) {
            let values = logits.narrow(1, offset, count)?.to_vec()?;
            let action = values
                .iter()
                .position_max_by(|a, b| a.total_cmp(b))
                .expect("validated categorical group must be nonempty");
            actions.push(action as f32);
        }
        Ok(Self::Tensor::from_vec_like(
            actions,
            [1, self.categories.len()],
            &logits,
        )?)
    }
}

impl<N: Network + ToSafetensors> ToSafetensors for MultiCategorical<N> {
    fn to_safetensors(&self) -> Result<Vec<u8>> {
        self.logits.to_safetensors()
    }
}

impl<N: Network> Policy2 for MultiCategorical<N> {
    fn action_shape(&self) -> Shape {
        [self.categories.len()].into()
    }

    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations)?;
        super::action_batch(&actions, logits.to_shape()[0], self.categories.len())?;
        let mut selected = Vec::with_capacity(self.categories.len());
        for (index, (offset, count)) in action_ranges(&self.categories).enumerate() {
            let log_probs = logits.narrow(1, offset, count)?.log_softmax(1)?;
            selected.push(log_probs.gather(1, &actions.narrow(1, index, 1)?)?);
        }
        Ok(Self::Tensor::add_multiple(&selected)?)
    }

    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations)?;
        let mut entropies = Vec::with_capacity(self.categories.len());
        for (offset, count) in action_ranges(&self.categories) {
            let logits = logits.narrow(1, offset, count)?;
            entropies.push(
                logits
                    .softmax(1)?
                    .mul(&logits.log_softmax(1)?)?
                    .neg()?
                    .sum_dim(1)?,
            );
        }
        Ok(Self::Tensor::add_multiple(&entropies)?.mean()?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
