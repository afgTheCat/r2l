use r2l_candle_lm::{
    CandleModuleWithValueFunction,
    distributions::DistributionKind,
    learning_module::{
        DecoupledActorCriticLM, ParalellActorCriticLM, PolicyValuesLosses, SequentialValueFunction,
    },
};
use r2l_core::policies::{LearningModule, ModuleWithValueFunction};

pub enum ActorCriticKind {
    Decoupled(DecoupledActorCriticLM),
    Paralell(ParalellActorCriticLM),
}

impl ActorCriticKind {
    pub fn policy_learning_rate(&self) -> f64 {
        match self {
            Self::Decoupled(decoupled) => decoupled.policy_learning_rate(),
            Self::Paralell(paralell) => paralell.policy_learning_rate(),
        }
    }

    pub fn set_grad_clipping(&mut self, max_grad_norm: Option<f32>) {
        match self {
            Self::Decoupled(decoupled) => decoupled.set_policy_grad_clip(max_grad_norm),
            Self::Paralell(paralell) => paralell.set_grad_clip(max_grad_norm),
        }
    }
}

impl LearningModule for ActorCriticKind {
    type Losses = PolicyValuesLosses;

    fn update(&mut self, losses: Self::Losses) -> anyhow::Result<()> {
        match self {
            Self::Decoupled(lm) => lm.update(losses),
            Self::Paralell(lm) => lm.update(losses),
        }
    }
}

pub struct R2lCandleLearningModule {
    pub policy: DistributionKind,
    pub learning_module: ActorCriticKind,
    pub value_function: SequentialValueFunction,
}

impl R2lCandleLearningModule {
    pub fn set_grad_clipping(&mut self, gradient_clipping: Option<f32>) {
        self.learning_module.set_grad_clipping(gradient_clipping);
    }

    pub fn policy_learning_rate(&self) -> f64 {
        self.learning_module.policy_learning_rate()
    }
}

impl ModuleWithValueFunction for R2lCandleLearningModule {
    type Tensor = candle_core::Tensor;
    type InferenceTensor = candle_core::Tensor;
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
}

impl CandleModuleWithValueFunction for R2lCandleLearningModule {}

// TODO: add the builder here
