use crate::sequential::Sequential;
use burn::{
    module::{AutodiffModule, Module, ModuleDisplay},
    optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_core::{
    distributions::Distribution,
    policies::{LearningModule, ValueFunction},
};

pub struct PolicyValuesLosses<B: AutodiffBackend> {
    pub policy_loss: Tensor<B, 1>,
    pub value_loss: Tensor<B, 1>,
}

// a model with a value function
#[derive(Debug, Module)]
pub struct ParalellActorModel<B: Backend, M: Module<B>> {
    pub distr: M,
    pub value_net: Sequential<B>,
}

impl<B: Backend, M: Module<B>> ParalellActorModel<B, M> {
    pub fn new(distr: M, value_net: Sequential<B>) -> Self {
        Self { distr, value_net }
    }
}

// TODO: this is messy, like mega messy.
pub struct ParalellActorCriticLM<
    B: AutodiffBackend,
    M: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>,
> where
    <M as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    pub model: ParalellActorModel<B, M>,
    pub optimizer: OptimizerAdaptor<AdamW, ParalellActorModel<B, M>, B>,
}

impl<B: AutodiffBackend, M: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>>
    ParalellActorCriticLM<B, M>
where
    <M as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    pub fn new(
        model: ParalellActorModel<B, M>,
        optimizer: OptimizerAdaptor<AdamW, ParalellActorModel<B, M>, B>,
    ) -> Self {
        Self { model, optimizer }
    }
}

impl<B: AutodiffBackend, M: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>>
    LearningModule for ParalellActorCriticLM<B, M>
where
    <M as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    type Losses = PolicyValuesLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        let loss = losses.policy_loss + losses.value_loss;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        let new_model = self.optimizer.step(1e-4, self.model.clone(), grads);
        self.model = new_model;
        Ok(())
    }
}

impl<B: AutodiffBackend, M: AutodiffModule<B> + ModuleDisplay + Distribution<Tensor = Tensor<B, 1>>>
    ValueFunction for ParalellActorCriticLM<B, M>
where
    <M as AutodiffModule<B>>::InnerModule: ModuleDisplay,
{
    type Tensor = Tensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let observation: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let value = self.model.value_net.forward(observation);
        Ok(value.squeeze(0))
    }
}
