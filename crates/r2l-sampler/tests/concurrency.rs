use std::{sync::Arc, thread, time::Duration};

use crossbeam::channel::{Receiver, Sender, bounded};
use r2l_core::{
    env::{Env, EnvBuilder, EnvDescription, Snapshot, Space},
    error::Error,
    models::Actor,
    on_policy::algorithm::Sampler,
    tensor::VecTensor,
};
use r2l_sampler::{
    DirectSampler, DirectSamplerCore, DirectSamplerHook, RolloutMode, SamplerExecutionMode,
    SamplerHookResult,
};

struct CoordinatedEnv {
    entered: Sender<()>,
    release: Receiver<()>,
}

impl Env for CoordinatedEnv {
    type Tensor = VecTensor;

    fn reset(&mut self, _seed: u64) -> Result<Self::Tensor, Error> {
        Ok(VecTensor::from_vec(vec![0.0]))
    }

    fn step(&mut self, _action: Self::Tensor) -> Result<Snapshot<Self::Tensor>, Error> {
        self.entered.send(()).unwrap();
        self.release.recv().unwrap();
        Ok(Snapshot::new(
            VecTensor::from_vec(vec![1.0]),
            1.0,
            false,
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

#[derive(Clone)]
struct ActorStub;

impl Actor for ActorStub {
    type Tensor = VecTensor;

    fn action(&self, _observation: Self::Tensor) -> Result<Self::Tensor, Error> {
        Ok(VecTensor::from_vec(vec![0.0]))
    }

    fn mode_action(&self, observation: Self::Tensor) -> Result<Self::Tensor, Error> {
        self.action(observation)
    }
}

struct OneStep(bool);

impl DirectSamplerHook for OneStep {
    type E = CoordinatedEnv;

    fn hook(&mut self, _core: &mut DirectSamplerCore<Self::E>) -> SamplerHookResult {
        if self.0 {
            SamplerHookResult::Stop
        } else {
            self.0 = true;
            SamplerHookResult::Bound(RolloutMode::StepBound { n_steps: 1 })
        }
    }
}

#[test]
fn multithreaded_workers_enter_environment_steps_concurrently() {
    let (entered_tx, entered_rx) = bounded(2);
    let (release_tx, release_rx) = bounded(2);
    let builder: Arc<dyn EnvBuilder<Env = CoordinatedEnv>> = Arc::new(move || {
        Ok(CoordinatedEnv {
            entered: entered_tx.clone(),
            release: release_rx.clone(),
        })
    });
    let mut sampler = DirectSampler::build_from_env_builder(
        builder,
        2,
        OneStep(false),
        SamplerExecutionMode::MultiThreaded,
    )
    .unwrap();

    let collect = thread::spawn(move || {
        let result = sampler.collect_rollouts(ActorStub);
        sampler.shutdown();
        result
    });

    let first = entered_rx.recv_timeout(Duration::from_secs(2));
    let second = entered_rx.recv_timeout(Duration::from_secs(2));
    release_tx.send(()).unwrap();
    release_tx.send(()).unwrap();

    assert!(first.is_ok(), "no worker entered Env::step");
    assert!(
        second.is_ok(),
        "workers did not execute Env::step concurrently"
    );
    collect.join().unwrap().unwrap();
}
