use std::marker::PhantomData;

use burn::tensor::{Tensor as BurnTensor, backend::AutodiffBackend};
use r2l_agents::ppo2::{NewPPO, PPOModule2, RolloutLearningModule};
use r2l_burn_lm::learning_module::{BurnPolicy, ParalellActorCriticLM, PolicyValuesLosses};
use r2l_core::policies::{LearningModule, ValueFunction};

// TODO: finish this in some form
pub struct R2lBurnLearningModule<
    B: AutodiffBackend,
    D: BurnPolicy<B>,
    V: ValueFunction<Tensor = BurnTensor<B, 1>>,
> {
    lm: ParalellActorCriticLM<B, D>,
    _phantom: PhantomData<(B, D, V)>,
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, V: ValueFunction<Tensor = BurnTensor<B, 1>>>
    LearningModule for R2lBurnLearningModule<B, D, V>
{
    type Losses = PolicyValuesLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        self.lm.update(losses)
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, V: ValueFunction<Tensor = BurnTensor<B, 1>>>
    ValueFunction for R2lBurnLearningModule<B, D, V>
{
    type Tensor = BurnTensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        self.lm.calculate_values(observations)
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, V: ValueFunction<Tensor = BurnTensor<B, 1>>>
    RolloutLearningModule for R2lBurnLearningModule<B, D, V>
{
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

impl<B: AutodiffBackend, D: BurnPolicy<B>, V: ValueFunction<Tensor = BurnTensor<B, 1>>> PPOModule2
    for R2lBurnLearningModule<B, D, V>
{
}
