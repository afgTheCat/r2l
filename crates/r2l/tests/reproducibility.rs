use std::collections::BTreeMap;

use r2l::{
    Env, EnvDescription, PPOBuilder, SamplerExecutionMode, Snapshot, Space, TrainingLimit,
    VecTensor,
};
use r2l_core::{error::Result, models::ToSafetensors, tensor::R2lTensor};

struct SeededEnv;

impl Env for SeededEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<VecTensor> {
        Ok(VecTensor::from_vec(vec![1.0, -0.5]))
    }

    fn step(&mut self, action: VecTensor) -> Result<Snapshot<VecTensor>> {
        Ok(Snapshot::new(
            VecTensor::from_vec(vec![0.5, 1.0]),
            action.to_vec()?[0],
            true,
            false,
        ))
    }

    fn env_description(&self) -> EnvDescription<VecTensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: [2].into(),
            },
            Space::Discrete(2),
        )
    }
}

fn weights(actor: &impl ToSafetensors) -> BTreeMap<String, Vec<u8>> {
    let bytes = actor.to_safetensors().unwrap();
    safetensors::SafeTensors::deserialize(&bytes)
        .unwrap()
        .tensors()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.data().to_vec()))
        .collect()
}

// Keep seeded Burn runs in a single test: its backend RNG is shared within a process.
#[test]
fn seeded_builds_and_training_remain_reproducible() {
    macro_rules! train {
        ($seed:expr, $($backend:ident)?) => {{
            let mut algorithm = PPOBuilder::new(|| Ok(SeededEnv), 1).unwrap()
                $(.$backend())?
                .with_execution_mode(SamplerExecutionMode::SingleThreaded)
                .with_rollout_steps(4)
                .with_training_limit(TrainingLimit::rollouts(2))
                .with_policy_hidden_layers(vec![4])
                .with_value_hidden_layers(vec![4])
                .with_sample_size(2)
                .with_total_epochs(2)
                .with_learning_rate(0.01)
                .with_log_progress(false)
                .with_seed($seed)
                .build().unwrap();
            let before = weights(&algorithm.runtime.actor());
            algorithm.train().unwrap();
            let after = weights(&algorithm.runtime.actor());
            assert_ne!(before, after);
            (before, after)
        }};
    }
    let candle = train!(7,);
    assert_eq!(candle, train!(7,));
    assert_ne!(candle.0, train!(8,).0);
    #[cfg(not(feature = "simd"))]
    {
        let burn = train!(7, with_burn);
        assert_eq!(burn, train!(7, with_burn));
        assert_ne!(burn.0, train!(8, with_burn).0);
    }
}
