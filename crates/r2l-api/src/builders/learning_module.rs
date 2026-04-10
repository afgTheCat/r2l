use burn::{optim::AdamWConfig, tensor::backend::AutodiffBackend};
use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use r2l_burn::{
    learning_module::{
        BurnActorCriticLMKind, BurnDecoupledActorCriticLM, BurnParalellActorCriticLM, BurnPolicy,
        ParalellActorModel,
    },
    sequential::Sequential,
};
use r2l_candle::{
    learning_module::{
        CandleActorCriticKind, DecoupledActorCriticLM, ParalellActorCriticLM,
        SequentialValueFunction,
    },
    optimizer::OptimizerWithMaxGrad,
    thread_safe_sequential::build_sequential,
};

pub enum LearningModuleType {
    Paralell {
        value_layers: Vec<usize>,
        max_grad_norm: Option<f32>,
    },
    Decoupled {
        value_layers: Vec<usize>,
        policy_max_grad_norm: Option<f32>,
        value_max_grad_norm: Option<f32>,
    },
}

// TODO: we probably need to rethink this. We need to add the optimizer params
pub struct LearningModuleBuilder {
    pub learning_module_type: LearningModuleType,
    pub params: ParamsAdamW,
}

impl LearningModuleBuilder {
    pub fn build_candle(
        &self,
        distribution_varmap: VarMap,
        distr_var_builder: VarBuilder,
        observation_size: usize,
        device: &Device,
    ) -> Result<(SequentialValueFunction, CandleActorCriticKind)> {
        match &self.learning_module_type {
            LearningModuleType::Paralell {
                value_layers,
                max_grad_norm,
            } => {
                let value_layers = &[&value_layers[..], &[1]].concat();
                let value_net =
                    build_sequential(observation_size, value_layers, &distr_var_builder, "value")?;
                let optimizer = AdamW::new(distribution_varmap.all_vars(), self.params.clone())?;
                let optimizer_with_grad =
                    OptimizerWithMaxGrad::new(optimizer, *max_grad_norm, distribution_varmap);
                Ok((
                    SequentialValueFunction { value_net },
                    CandleActorCriticKind::Paralell(ParalellActorCriticLM {
                        optimizer_with_grad,
                    }),
                ))
            }
            LearningModuleType::Decoupled {
                value_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
            } => {
                let critic_varmap = VarMap::new();
                let critic_vb = VarBuilder::from_varmap(&critic_varmap, DType::F32, device);
                let value_layers = &[&value_layers[..], &[1]].concat();
                let value_net =
                    build_sequential(observation_size, value_layers, &critic_vb, "value")?;
                let policy_optimizer =
                    AdamW::new(distribution_varmap.all_vars(), self.params.clone())?;
                let value_optimizer = AdamW::new(critic_varmap.all_vars(), self.params.clone())?;
                let policy_optimizer_with_grad = OptimizerWithMaxGrad::new(
                    policy_optimizer,
                    *policy_max_grad_norm,
                    distribution_varmap,
                );
                let value_optimizer_with_grad =
                    OptimizerWithMaxGrad::new(value_optimizer, *value_max_grad_norm, critic_varmap);
                Ok((
                    SequentialValueFunction { value_net },
                    CandleActorCriticKind::Decoupled(DecoupledActorCriticLM {
                        policy_optimizer_with_grad,
                        value_optimizer_with_grad,
                    }),
                ))
            }
        }
    }

    pub fn build_burn<B: AutodiffBackend, D: BurnPolicy<B>>(
        &self,
        observation_size: usize,
        distr: D,
    ) -> BurnActorCriticLMKind<B, D> {
        match &self.learning_module_type {
            LearningModuleType::Paralell {
                value_layers,
                // TODO: should we consider this
                max_grad_norm: _max_grad_norm,
            } => {
                let value_layers = &[&[observation_size][..], &value_layers[..], &[1]].concat();
                let value_net: Sequential<B> = Sequential::build(value_layers);
                let model = ParalellActorModel::new(distr, value_net);
                BurnActorCriticLMKind::Paralell(BurnParalellActorCriticLM::new(
                    model,
                    AdamWConfig::new()
                        .with_beta_1(self.params.beta1 as f32)
                        .with_beta_2(self.params.beta2 as f32)
                        .with_epsilon(self.params.eps as f32)
                        .with_weight_decay(self.params.weight_decay as f32)
                        .init(),
                    self.params.lr,
                ))
            }
            LearningModuleType::Decoupled {
                value_layers,
                // TODO: should we consider this
                policy_max_grad_norm: _policy_max_grad_norm,
                value_max_grad_norm: _value_max_grad_norm,
            } => {
                let value_layers = &[&[observation_size][..], &value_layers[..], &[1]].concat();
                let value_net: Sequential<B> = Sequential::build(value_layers);
                BurnActorCriticLMKind::Decoupled(BurnDecoupledActorCriticLM::new(
                    distr,
                    value_net,
                    AdamWConfig::new()
                        .with_beta_1(self.params.beta1 as f32)
                        .with_beta_2(self.params.beta2 as f32)
                        .with_epsilon(self.params.eps as f32)
                        .with_weight_decay(self.params.weight_decay as f32)
                        .init(),
                    self.params.lr,
                    AdamWConfig::new()
                        .with_beta_1(self.params.beta1 as f32)
                        .with_beta_2(self.params.beta2 as f32)
                        .with_epsilon(self.params.eps as f32)
                        .with_weight_decay(self.params.weight_decay as f32)
                        .init(),
                    self.params.lr,
                ))
            }
        }
    }
}
