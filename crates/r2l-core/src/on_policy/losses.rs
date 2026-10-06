use crate::{
    error::{Error, Result},
    tensor::R2lTensor,
};

/// Reduced policy/value losses shared by on-policy learners.
pub struct PolicyValueLosses<T> {
    /// Policy loss, including any entropy term.
    pub policy_loss: T,
    /// Value-function loss.
    pub value_loss: T,
    /// Multiplier applied to the value loss during optimization.
    pub vf_coeff: f32,
}

impl<T> PolicyValueLosses<T> {
    /// Creates a bundle of single-element losses with a value multiplier of one.
    #[must_use]
    pub fn new(policy_loss: T, value_loss: T) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: 1.0,
        }
    }

    /// Sets the value-loss multiplier applied during optimization.
    pub fn set_vf_coeff(&mut self, vf_coeff: f32) {
        self.vf_coeff = vf_coeff;
    }
}

impl<T: R2lTensor> PolicyValueLosses<T> {
    /// Adds an already signed and weighted entropy term to the policy loss.
    ///
    /// # Errors
    /// Returns an error unless both terms contain one element and can be added.
    pub fn add_entropy_loss(&mut self, entropy_loss: &T) -> Result<()> {
        if self.policy_loss.size() != 1 || entropy_loss.size() != 1 {
            return Err(Error::invalid_parameter(
                "entropy loss",
                "single-element policy and entropy losses",
                format!(
                    "{} and {} elements",
                    self.policy_loss.size(),
                    entropy_loss.size()
                ),
            ));
        }
        // Single-element tensors may have different ranks; mean gives each
        // backend its scalar representation without breaking the gradient graph.
        self.policy_loss = self.policy_loss.mean()?.add(&entropy_loss.mean()?)?;
        Ok(())
    }
}
