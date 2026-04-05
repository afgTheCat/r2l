use burn::{
    grad_clipping::GradientClipping,
    tensor::{Tensor as BurnTensor, backend::AutodiffBackend},
};
use r2l_agents::ppo2::{NewPPO, PPOModule2, RolloutLearningModule};
use r2l_burn_lm::learning_module::{BurnPolicy, ParalellActorCriticLM, PolicyValuesLosses};
use r2l_core::policies::{LearningModule, ValueFunction};

use crate::hooks::ppo::PPOHook;

// TODO: finish this in some form
pub struct R2lBurnLearningModule<B: AutodiffBackend, D: BurnPolicy<B>> {
    lm: ParalellActorCriticLM<B, D>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> R2lBurnLearningModule<B, D> {
    pub fn set_grad_clipping(&mut self, gradient_clipping: GradientClipping) {
        self.lm.set_grad_clipping(gradient_clipping);
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> LearningModule for R2lBurnLearningModule<B, D> {
    type Losses = PolicyValuesLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        self.lm.update(losses)
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> ValueFunction for R2lBurnLearningModule<B, D> {
    type Tensor = BurnTensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        self.lm.calculate_values(observations)
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> RolloutLearningModule for R2lBurnLearningModule<B, D> {
    type LearningTensor = BurnTensor<B, 1>;
    type InferenceTensor = BurnTensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.lm.model.distr.valid()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.lm.model.distr
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        BurnTensor::from_data(slice, &Default::default())
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        BurnTensor::from_data(t.to_data(), &Default::default())
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> PPOModule2 for R2lBurnLearningModule<B, D> {}

pub struct BurnPPO<B: AutodiffBackend, D: BurnPolicy<B>>(
    pub NewPPO<R2lBurnLearningModule<B, D>, PPOHook<R2lBurnLearningModule<B, D>>>,
);
