use std::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{Tensor as BurnTensor, backend::AutodiffBackend},
};
use r2l_agents::ppo2::{NewPPO, PPOModule2};
use r2l_burn_lm::learning_module::{BurnPolicy, ParalellActorCriticLM, PolicyValuesLosses};
use r2l_core::policies::ValueFunction;

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
    type Tensor = BurnTensor<B, 1>;
    type InferenceTensor = BurnTensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;
    type ValueFunction = V;
    type Losses = PolicyValuesLosses<B>;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        todo!()
    }

    fn get_policy(&self) -> &Self::Policy {
        todo!()
    }

    fn value_func(&self) -> &Self::ValueFunction {
        todo!()
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::Tensor {
        todo!()
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::Tensor {
        todo!()
    }

    fn get_losses(policy_loss: Self::Tensor, value_loss: Self::Tensor) -> Self::Losses {
        todo!()
    }

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        todo!()
    }
}

// pub struct R2lBurnLearningModule(pub NewPPO<>)
