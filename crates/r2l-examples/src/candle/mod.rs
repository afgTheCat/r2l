pub mod new_ppo;

use crate::PPOProgress;
use candle_core::{DType, Tensor};

impl PPOProgress {
    pub fn collect_batch_data(
        &mut self,
        ratio: &Tensor,
        entropy_loss: &Tensor,
        value_loss: &Tensor,
        policy_loss: &Tensor,
    ) -> candle_core::Result<()> {
        let clip_fraction = (ratio - 1.)?
            .abs()?
            .gt(self.clip_range)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        self.clip_fractions.push(clip_fraction);
        self.entropy_losses.push(entropy_loss.to_scalar()?);
        self.value_losses.push(value_loss.to_scalar()?);
        self.policy_losses.push(policy_loss.to_scalar()?);
        Ok(())
    }
}
