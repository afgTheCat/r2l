use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use r2l::{Env, EnvDescription, PPOBuilder, Snapshot, Space, TrainingLimit, VecTensor};
use r2l_core::error::Error;

struct DropTrackedEnv {
    drops: Arc<AtomicUsize>,
}

impl Drop for DropTrackedEnv {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

impl Env for DropTrackedEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor, Error> {
        Ok(VecTensor::from_vec(vec![0.0]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>, Error> {
        Ok(Snapshot::new(
            VecTensor::from_vec(vec![0.0]),
            1.0,
            true,
            false,
        ))
    }

    fn env_description(&self) -> EnvDescription<Self::Tensor> {
        EnvDescription::new(
            Space::Box {
                min: None,
                max: None,
                shape: vec![1],
            },
            Space::Discrete(1),
        )
    }
}

#[test]
fn training_completion_keeps_sampler_resources_alive_until_algorithm_drop() {
    let drops = Arc::new(AtomicUsize::new(0));
    let env_builder = {
        let drops = drops.clone();
        move || {
            Ok(DropTrackedEnv {
                drops: drops.clone(),
            })
        }
    };
    let mut algorithm = PPOBuilder::new(env_builder, 1)
        .unwrap()
        .with_rollout_steps(2)
        .with_training_limit(TrainingLimit::rollouts(1))
        .with_policy_hidden_layers(vec![2])
        .with_sample_size(2)
        .with_total_epochs(1)
        .with_seed(5)
        .build()
        .unwrap();
    let drops_after_build = drops.load(Ordering::SeqCst);

    algorithm.train().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), drops_after_build);

    drop(algorithm);
    assert_eq!(drops.load(Ordering::SeqCst), drops_after_build + 1);
}

#[test]
fn stop_training_command_breaks_the_loop_without_destroying_the_algorithm() {
    let drops = Arc::new(AtomicUsize::new(0));
    let env_builder = {
        let drops = drops.clone();
        move || {
            Ok(DropTrackedEnv {
                drops: drops.clone(),
            })
        }
    };
    let builder = PPOBuilder::new(env_builder, 1)
        .unwrap()
        .with_rollout_steps(1)
        .with_training_limit(TrainingLimit::rollouts(1_000_000))
        .with_policy_hidden_layers(vec![2])
        .with_sample_size(1)
        .with_total_epochs(1)
        .with_seed(5);
    let (builder, control) = builder.with_control();
    let mut algorithm = builder.build().unwrap();
    let drops_after_build = drops.load(Ordering::SeqCst);

    let training = std::thread::spawn(move || {
        algorithm.train().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), drops_after_build);
        algorithm
    });
    control.stop_training().unwrap();
    let algorithm = training.join().unwrap();

    drop(algorithm);
}
