use crate::builders::{
    distribution::{DistributionBuilder, DistributionType},
    learning_module::{LearningModuleBuilder, LearningModuleType},
};
use candle_core::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};
use r2l_agents::ppo::{EmptyPPO3Hooks, PPO};
use r2l_core::{
    distributions::DistributionKind, env::EnvironmentDescription,
    policies::learning_modules::LearningModuleKind,
};

pub struct PPOBuilder {
    pub distribution_builder: DistributionBuilder,
    pub learning_module_builder: LearningModuleBuilder,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for PPOBuilder {
    fn default() -> Self {
        PPOBuilder {
            distribution_builder: DistributionBuilder {
                hidden_layers: vec![64, 64],
                distribution_type: DistributionType::Dynamic,
            },
            learning_module_builder: LearningModuleBuilder {
                learning_module_type: LearningModuleType::Paralell {
                    value_layers: vec![64, 64],
                    max_grad_norm: None,
                },
            },
            clip_range: 0.2,
            lambda: 0.8,
            gamma: 0.98,
            sample_size: 64,
        }
    }
}

impl PPOBuilder {
    pub fn build(
        &self,
        device: &Device,
        env_description: &EnvironmentDescription,
    ) -> Result<PPO<DistributionKind, LearningModuleKind>> {
        let distribution_varmap = VarMap::new();
        let distribution_var_builder =
            VarBuilder::from_varmap(&distribution_varmap, DType::F32, &device);
        let distribution =
            self.distribution_builder
                .build(&distribution_var_builder, device, env_description)?;
        let learning_module = self.learning_module_builder.build(
            distribution_varmap,
            distribution_var_builder,
            env_description,
            device,
        )?;
        Ok(PPO {
            distribution,
            learning_module,
            hooks: Box::new(EmptyPPO3Hooks),
            clip_range: self.clip_range,
            device: device.clone(),
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size,
        })
    }
}

// pub struct PPOBuilder2 {
//     pub distribution_kind: PPODistributionKind,
//     pub policy_builder: PolicyBuilder,
//     pub clip_range: f32,
//     pub gamma: f32,
//     pub lambda: f32,
//     pub sample_size: usize,
// }
//
// impl Default for PPOBuilder2 {
//     fn default() -> Self {
//         PPOBuilder2 {
//             distribution_kind: PPODistributionKind::Dynamic {
//                 hidden_layers: vec![64, 64],
//             },
//             policy_builder: PolicyBuilder {
//                 policy_type: PolicyType::Paralell {
//                     value_layers: vec![64, 64],
//                     max_grad_norm: None,
//                 },
//             },
//             clip_range: 0.2,
//             lambda: 0.8,
//             gamma: 0.98,
//             sample_size: 64,
//         }
//     }
// }
//
// impl PPOBuilder2 {
//     pub fn build(
//         &self,
//         device: &Device,
//         env_description: &EnvironmentDescription,
//     ) -> Result<PPO2<PolicyKind>> {
//         let policy =
//             self.policy_builder
//                 .build_policy(&self.distribution_kind, env_description, &device)?;
//         Ok(PPO2 {
//             policy,
//             hooks: Box::new(DefaultPPOHooks::default()),
//             clip_range: self.clip_range,
//             device: device.clone(),
//             gamma: self.gamma,
//             lambda: self.lambda,
//             sample_size: self.sample_size,
//         })
//     }
// }
