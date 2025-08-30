use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use r2l_candle_lm::{
    learning_module::{DecoupledActorCriticLM, LearningModuleKind, ParalellActorCriticLM},
    optimizer::OptimizerWithMaxGrad,
    thread_safe_sequential::build_sequential,
};
use r2l_core::env::EnvironmentDescription;

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

// TODO: we probably need to rethink this
pub struct LearningModuleBuilder {
    pub learning_module_type: LearningModuleType,
}

impl LearningModuleBuilder {
    pub fn build(
        &self,
        distribution_varmap: VarMap,
        distr_var_builder: VarBuilder,
        env_description: &EnvironmentDescription,
        device: &Device,
    ) -> Result<LearningModuleKind> {
        let input_size = env_description.observation_size();
        let optimizer_params = ParamsAdamW {
            lr: 3e-4,
            weight_decay: 0.01,
            ..Default::default()
        };
        match &self.learning_module_type {
            LearningModuleType::Paralell {
                value_layers,
                max_grad_norm,
            } => {
                let value_layers = &[&value_layers[..], &[1]].concat();
                let value_net =
                    build_sequential(input_size, value_layers, &distr_var_builder, "value")?;
                let optimizer =
                    AdamW::new(distribution_varmap.all_vars(), optimizer_params.clone())?;
                let optimizer_with_grad =
                    OptimizerWithMaxGrad::new(optimizer, *max_grad_norm, distribution_varmap);
                Ok(LearningModuleKind::Paralell(ParalellActorCriticLM {
                    value_net,
                    optimizer_with_grad,
                }))
            }
            LearningModuleType::Decoupled {
                value_layers,
                policy_max_grad_norm,
                value_max_grad_norm,
            } => {
                let critic_varmap = VarMap::new();
                let critic_vb = VarBuilder::from_varmap(&critic_varmap, DType::F32, &device);
                let value_layers = &[&value_layers[..], &[1]].concat();
                let value_net = build_sequential(input_size, value_layers, &critic_vb, "value")?;
                let policy_optimizer =
                    AdamW::new(distribution_varmap.all_vars(), optimizer_params.clone())?;
                let value_optimizer = AdamW::new(critic_varmap.all_vars(), optimizer_params)?;
                let policy_optimizer_with_grad = OptimizerWithMaxGrad::new(
                    policy_optimizer,
                    *policy_max_grad_norm,
                    distribution_varmap,
                );
                let value_optimizer_with_grad =
                    OptimizerWithMaxGrad::new(value_optimizer, *value_max_grad_norm, critic_varmap);
                Ok(LearningModuleKind::Decoupled(DecoupledActorCriticLM {
                    value_net,
                    policy_optimizer_with_grad,
                    value_optimizer_with_grad,
                }))
            }
        }
    }
}
