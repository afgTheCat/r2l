use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{Tensor as BurnTensor, backend::AutodiffBackend},
};
use r2l_agents::{
    ppo2::{NewPPO, PPOModule2},
};
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

impl<B: AutodiffBackend, D: BurnPolicy<B>, V: ValueFunction<Tensor = BurnTensor<B, 1>>> PPOModule2
    for R2lBurnLearningModule<B, D, V>
{
    type LearningTensor = BurnTensor<B, 1>;
    type InferenceTensor = BurnTensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        todo!()
    }

    fn get_policy(&self) -> &Self::Policy {
        todo!()
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        todo!()
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        todo!()
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, V: ValueFunction<Tensor = BurnTensor<B, 1>>> LearningModule
    for R2lBurnLearningModule<B, D, V>
{
    type Losses = PolicyValuesLosses<B>;

    fn update(&mut self, _losses: Self::Losses) -> anyhow::Result<()> {
        todo!()
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>, V: ValueFunction<Tensor = BurnTensor<B, 1>>> ValueFunction
    for R2lBurnLearningModule<B, D, V>
{
    type Tensor = BurnTensor<B, 1>;

    fn calculate_values(&self, _observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        todo!()
    }
}

// pub struct R2lBurnLearningModule(pub NewPPO<>)
