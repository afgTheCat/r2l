use crate::builders::{
    distribution::{DistributionBuilder, DistributionType},
    learning_module::{LearningModuleBuilder, LearningModuleType},
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use r2l_agents::a2c::{A2C, DefaultA2CHooks};
use r2l_candle_lm::{distributions::DistributionKind, learning_module::LearningModuleKind};
use r2l_core::env::EnvironmentDescription;

pub struct A2CBuilder {
    pub distribution_builder: DistributionBuilder,
    pub learning_module_builder: LearningModuleBuilder,
    pub clip_range: f32,
    pub gamma: f32,
    pub lambda: f32,
    pub sample_size: usize,
}

impl Default for A2CBuilder {
    fn default() -> Self {
        A2CBuilder {
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

impl A2CBuilder {
    pub fn build<T>(
        &self,
        device: &Device,
        env_description: &EnvironmentDescription<T>,
    ) -> Result<A2C<DistributionKind, LearningModuleKind>> {
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
        Ok(A2C {
            distribution,
            learning_module,
            hooks: Box::new(DefaultA2CHooks),
            device: device.clone(),
            gamma: self.gamma,
            lambda: self.lambda,
            sample_size: self.sample_size,
        })
    }
}
