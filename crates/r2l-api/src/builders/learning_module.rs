use candle_core::{DType, Device, Result};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use r2l_agents::candle_agents::ActorCriticKind;
use r2l_candle_lm::{
    learning_module::{DecoupledActorCriticLM, ParalellActorCriticLM, SequentialValueFunction},
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

// TODO: we probably need to rethink this. We need to add the optimizer params
pub struct LearningModuleBuilder {
    pub learning_module_type: LearningModuleType,
    pub observation_size: Option<usize>,
}

impl LearningModuleBuilder {
    pub fn build(
        &self,
        distribution_varmap: VarMap,
        distr_var_builder: VarBuilder,
        device: &Device,
    ) -> Result<(SequentialValueFunction, ActorCriticKind)> {
        let observation_size = self.observation_size.unwrap();
        let optimizer_params = ParamsAdamW {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-5,
            weight_decay: 1e-4,
        };
        match &self.learning_module_type {
            LearningModuleType::Paralell {
                value_layers,
                max_grad_norm,
            } => {
                let value_layers = &[&value_layers[..], &[1]].concat();
                let value_net =
                    build_sequential(observation_size, value_layers, &distr_var_builder, "value")?;
                let optimizer =
                    AdamW::new(distribution_varmap.all_vars(), optimizer_params.clone())?;
                let optimizer_with_grad =
                    OptimizerWithMaxGrad::new(optimizer, *max_grad_norm, distribution_varmap);
                Ok((
                    SequentialValueFunction { value_net },
                    ActorCriticKind::Paralell(ParalellActorCriticLM {
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
                    AdamW::new(distribution_varmap.all_vars(), optimizer_params.clone())?;
                let value_optimizer = AdamW::new(critic_varmap.all_vars(), optimizer_params)?;
                let policy_optimizer_with_grad = OptimizerWithMaxGrad::new(
                    policy_optimizer,
                    *policy_max_grad_norm,
                    distribution_varmap,
                );
                let value_optimizer_with_grad =
                    OptimizerWithMaxGrad::new(value_optimizer, *value_max_grad_norm, critic_varmap);
                Ok((
                    SequentialValueFunction { value_net },
                    ActorCriticKind::Decoupled(DecoupledActorCriticLM {
                        policy_optimizer_with_grad,
                        value_optimizer_with_grad,
                    }),
                ))
            }
        }
    }

    // I gues we don't really need this
    pub fn build_with_env<T>(
        &mut self,
        distribution_varmap: VarMap,
        distr_var_builder: VarBuilder,
        env_description: &EnvironmentDescription<T>,
        device: &Device,
    ) -> Result<(SequentialValueFunction, ActorCriticKind)> {
        self.observation_size = Some(env_description.observation_size());
        self.build(distribution_varmap, distr_var_builder, device)
    }
}
