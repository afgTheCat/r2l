use crate::learning_module::R2lCandleLearningModule;
use candle_core::Tensor;
use r2l_agents::ppo2::PPOModule2;
use r2l_candle_lm::{
    distributions::DistributionKind,
    learning_module::{PolicyValuesLosses, SequentialValueFunction},
};
use r2l_core::policies::LearningModule;

impl PPOModule2 for R2lCandleLearningModule {
    type Tensor = Tensor;
    type InferenceTensor = Tensor;
    type Policy = DistributionKind;
    type InferencePolicy = DistributionKind;
    type ValueFunction = SequentialValueFunction;
    type Losses = PolicyValuesLosses;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.policy.clone()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        self.learning_module.update(losses)
    }

    fn value_func(&self) -> &Self::ValueFunction {
        &self.value_function
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::Tensor {
        Tensor::from_slice(slice, slice.len(), &candle_core::Device::Cpu).unwrap()
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::Tensor {
        t.clone()
    }

    fn get_losses(policy_loss: Self::Tensor, value_loss: Self::Tensor) -> Self::Losses {
        PolicyValuesLosses::new(policy_loss, value_loss)
    }
}
