use r2l_core::{
    Shape,
    error::{Error, Result},
    models::Actor,
    rng::with_rng,
    tensor::R2lTensor,
};
use rand_distr::{Bernoulli, Distribution};

use crate::{Network, Policy};

/// Independent binary actions, with one network logit per action bit.
#[derive(Debug, Clone)]
pub struct MultiBernoulli<N: Network> {
    pub(super) logits: N,
}

impl<N: Network> MultiBernoulli<N> {
    /// Builds a policy from a network with a positive output width.
    ///
    /// # Arguments
    /// * `logits` - Network producing one logit per action bit.
    ///
    /// # Errors
    /// Returns an error for an empty or non-vector output shape.
    pub fn new(logits: N) -> Result<Self> {
        super::output_width(&logits)?;
        Ok(Self { logits })
    }
}

impl<N: Network> Actor for MultiBernoulli<N> {
    type Tensor = N::Tensor;

    fn action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        let logits = self.logits.forward(observation)?;
        let actions = logits
            .log_sigmoid()?
            .exp()?
            .to_vec()?
            .into_iter()
            .map(|prob| {
                let distribution = Bernoulli::new(f64::from(prob)).map_err(Error::wrap)?;
                Ok(f32::from(with_rng(|rng| distribution.sample(rng))))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::Tensor::from_vec_like(
            actions,
            logits.to_shape(),
            &logits,
        )?)
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor> {
        super::single_observation(&observation)?;
        let logits = self.logits.forward(observation)?;
        let actions = logits
            .to_vec()?
            .iter()
            .map(|&value| f32::from(value >= 0.))
            .collect();
        Ok(Self::Tensor::from_vec_like(
            actions,
            logits.to_shape(),
            &logits,
        )?)
    }
}

impl<N: Network> Policy for MultiBernoulli<N> {
    fn action_shape(&self) -> Shape {
        self.logits.output_shape()
    }

    fn log_probs(&self, observations: Self::Tensor, actions: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations)?;
        let shape = logits.to_shape();
        super::action_batch(&actions, shape[0], shape[1])?;
        let log_p = logits.log_sigmoid()?;
        let log_q = logits.neg()?.log_sigmoid()?;
        Ok(actions
            .mul(&log_p)?
            .add(&actions.neg()?.add_scalar(1.)?.mul(&log_q)?)?
            .sum_dim(1)?)
    }

    fn entropy(&self, observations: Self::Tensor) -> Result<Self::Tensor> {
        let logits = self.logits.forward(observations)?;
        let log_p = logits.log_sigmoid()?;
        let log_q = logits.neg()?.log_sigmoid()?;
        Ok(log_p
            .exp()?
            .mul(&log_p)?
            .add(&log_q.exp()?.mul(&log_q)?)?
            .neg()?
            .sum_dim(1)?
            .mean()?)
    }

    fn std(&self) -> Result<Option<f32>> {
        Ok(None)
    }
}
