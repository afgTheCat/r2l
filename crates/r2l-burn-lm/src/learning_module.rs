use crate::sequential::Sequential;
use burn::{
    grad_clipping::{self, GradientClipping},
    module::{AutodiffModule, Module, ModuleDisplay},
    optim::{AdamW, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::Backend,
    tensor::{Tensor, backend::AutodiffBackend},
};
use r2l_core::{
    distributions::Policy,
    losses::PolicyValuesLosses,
    policies::{LearningModule, OnPolicyLearningModule, ValueFunction},
};

// A series constraints that we need for the policy to work nicely with AdamW
pub trait BurnPolicy<B: AutodiffBackend>:
    AutodiffModule<B, InnerModule: ModuleDisplay + Policy<Tensor = Tensor<B::InnerBackend, 1>>>
    + ModuleDisplay
    + Policy<Tensor = Tensor<B, 1>>
{
}

impl<B: AutodiffBackend, M> BurnPolicy<B> for M where
    M: AutodiffModule<B, InnerModule: ModuleDisplay + Policy<Tensor = Tensor<B::InnerBackend, 1>>>
        + ModuleDisplay
        + Policy<Tensor = Tensor<B, 1>>
{
}

pub struct BurnPolicyValuesLosses<B: AutodiffBackend> {
    pub policy_loss: Tensor<B, 1>,
    pub value_loss: Tensor<B, 1>,
    pub vf_coeff: Option<f32>,
}

impl<B: AutodiffBackend> PolicyValuesLosses<Tensor<B, 1>> for BurnPolicyValuesLosses<B> {
    fn losses(policy_loss: Tensor<B, 1>, value_loss: Tensor<B, 1>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }
}

impl<B: AutodiffBackend> BurnPolicyValuesLosses<B> {
    pub fn new(policy_loss: Tensor<B, 1>, value_loss: Tensor<B, 1>) -> Self {
        Self {
            policy_loss,
            value_loss,
            vf_coeff: None,
        }
    }

    pub fn apply_entropy(&mut self, entropy: Tensor<B, 1>) {
        self.policy_loss = self.policy_loss.clone() + entropy;
    }

    pub fn set_vf_coeff(&mut self, vf_coeff: Option<f32>) {
        self.vf_coeff = vf_coeff;
    }
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

pub struct ParalellActorCriticLM<B: AutodiffBackend, M: BurnPolicy<B>> {
    pub model: ParalellActorModel<B, M>,
    // NOTE:the optimizer needs to be optimizing both the distr and the value net at the same time
    pub optimizer: OptimizerAdaptor<AdamW, ParalellActorModel<B, M>, B>,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ParalellActorCriticLM<B, M> {
    pub fn new(
        model: ParalellActorModel<B, M>,
        optimizer: OptimizerAdaptor<AdamW, ParalellActorModel<B, M>, B>,
    ) -> Self {
        Self { model, optimizer }
    }

    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        self.optimizer = self.optimizer.clone().with_grad_clipping(grad_clipping);
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> LearningModule for ParalellActorCriticLM<B, M> {
    type Losses = BurnPolicyValuesLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        let loss = if let Some(vf_coeff) = losses.vf_coeff {
            losses.policy_loss + losses.value_loss.mul_scalar(vf_coeff)
        } else {
            losses.policy_loss + losses.value_loss
        };
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        let new_model = self.optimizer.step(3e-4, self.model.clone(), grads);
        self.model = new_model;
        Ok(())
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for ParalellActorCriticLM<B, M> {
    type Tensor = Tensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let observation: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let value = self.model.value_net.forward(observation);
        Ok(value.squeeze())
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for ParalellActorCriticLM<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.model.distr.valid()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.model.distr
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_data(slice, &Default::default())
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }
}

pub struct DecoupledActorCriticLM<B: AutodiffBackend, M: BurnPolicy<B>> {
    pub policy: M,
    pub value_net: Sequential<B>,
    pub policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
    pub value_net_optimizer: OptimizerAdaptor<AdamW, Sequential<B>, B>,
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> DecoupledActorCriticLM<B, M> {
    pub fn new(
        policy: M,
        value_net: Sequential<B>,
        policy_optimizer: OptimizerAdaptor<AdamW, M, B>,
        value_net_optimizer: OptimizerAdaptor<AdamW, Sequential<B>, B>,
    ) -> Self {
        Self {
            policy,
            value_net,
            policy_optimizer,
            value_net_optimizer,
        }
    }

    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        self.policy_optimizer = self
            .policy_optimizer
            .clone()
            .with_grad_clipping(grad_clipping);
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> LearningModule for DecoupledActorCriticLM<B, M> {
    type Losses = BurnPolicyValuesLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        let policy_grads = losses.policy_loss.backward();
        let policy_grads = GradientsParams::from_grads(policy_grads, &self.policy);
        // TODO: learning rate is hardcoded here
        self.policy = self
            .policy_optimizer
            .step(3e-4, self.policy.clone(), policy_grads);
        let value_loss = if let Some(vf_coeff) = losses.vf_coeff {
            losses.value_loss * vf_coeff
        } else {
            losses.value_loss
        };
        let value_grads = value_loss.backward();
        let value_grads = GradientsParams::from_grads(value_grads, &self.value_net);
        self.value_net = self
            .value_net_optimizer
            .step(3e-4, self.value_net.clone(), value_grads);
        Ok(())
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for DecoupledActorCriticLM<B, M> {
    type Tensor = Tensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        let observation: Tensor<B, 2> = Tensor::stack(observations.to_vec(), 0);
        let value = self.value_net.forward(observation);
        Ok(value.squeeze())
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for DecoupledActorCriticLM<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        self.policy.valid()
    }

    fn get_policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        Tensor::from_data(slice, &Default::default())
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }
}

pub enum ActorCriticLMKind<B: AutodiffBackend, D: BurnPolicy<B>> {
    Paralell(ParalellActorCriticLM<B, D>),
    Decoupled(DecoupledActorCriticLM<B, D>),
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> ActorCriticLMKind<B, D> {
    pub fn set_grad_clipping(&mut self, grad_clipping: GradientClipping) {
        match self {
            Self::Paralell(lm) => lm.set_grad_clipping(grad_clipping),
            Self::Decoupled(lm) => lm.set_grad_clipping(grad_clipping),
        }
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> LearningModule for ActorCriticLMKind<B, M> {
    type Losses = BurnPolicyValuesLosses<B>;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        match self {
            Self::Paralell(lm) => lm.update(losses),
            Self::Decoupled(lm) => lm.update(losses),
        }
    }
}

impl<B: AutodiffBackend, M: BurnPolicy<B>> ValueFunction for ActorCriticLMKind<B, M> {
    type Tensor = Tensor<B, 1>;

    fn calculate_values(&self, observations: &[Self::Tensor]) -> anyhow::Result<Self::Tensor> {
        match self {
            Self::Paralell(lm) => lm.calculate_values(observations),
            Self::Decoupled(lm) => lm.calculate_values(observations),
        }
    }
}

impl<B: AutodiffBackend, D: BurnPolicy<B>> OnPolicyLearningModule for ActorCriticLMKind<B, D> {
    type LearningTensor = Tensor<B, 1>;
    type InferenceTensor = Tensor<B::InnerBackend, 1>;
    type Policy = D;
    type InferencePolicy = D::InnerModule;

    fn get_inference_policy(&self) -> Self::InferencePolicy {
        match self {
            Self::Paralell(lm) => lm.get_inference_policy(),
            Self::Decoupled(lm) => lm.get_inference_policy(),
        }
    }

    fn get_policy(&self) -> &Self::Policy {
        match self {
            Self::Paralell(lm) => lm.get_policy(),
            Self::Decoupled(lm) => lm.get_policy(),
        }
    }

    fn tensor_from_slice(&self, slice: &[f32]) -> Self::LearningTensor {
        match self {
            Self::Paralell(lm) => lm.tensor_from_slice(slice),
            Self::Decoupled(lm) => lm.tensor_from_slice(slice),
        }
    }

    fn lifter(t: &Self::InferenceTensor) -> Self::LearningTensor {
        Tensor::from_data(t.to_data(), &Default::default())
    }
}
