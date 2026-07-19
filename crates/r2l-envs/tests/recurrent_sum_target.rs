use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamWConfig,
    tensor::{Tensor, backend::Backend},
};
use r2l_agents::on_policy_algorithms::{
    Advantages, Returns,
    recurrent_ppo::{RecurrentPPO, RecurrentPPOBatchData, RecurrentPPOHook, RecurrentPPOParams},
};
use r2l_burn::{
    distributions::recurrent_diagonal::RecurrentDiagGaussianDistribution,
    learning_module::PolicyValueModule,
};
use r2l_core::{
    HookResult,
    buffers::TrajectoryBatch,
    env::{Env, EnvBuilderType},
    models::{Actor, LearningModule},
    on_policy::{
        algorithm::{DefaultAdapter, OnPolicyRuntime},
        learning_module::OnPolicyLearningModule,
    },
    rng::set_seed,
    tensor::{R2lTensor, TensorData},
};
use r2l_envs::SumTargetEnv;
use r2l_sampler::{
    R2lSampler, R2lSamplerCore, RolloutMode, SamplerExecutionMode, SamplerHook, SamplerHookResult,
};

type InferenceBackend = NdArray<f32>;
type TrainBackend = Autodiff<InferenceBackend>;
type Policy = RecurrentDiagGaussianDistribution<TrainBackend>;
type Module = PolicyValueModule<TrainBackend, Policy>;

const EPISODE_STEPS: usize = 16;
const N_ENVS: usize = 4;
const STEPS_PER_ROLLOUT: usize = EPISODE_STEPS * 2;
const TRAINING_ROLLOUTS: usize = 64;
const PPO_EPOCHS: usize = 2;

struct StepHook {
    collect: bool,
}

impl SamplerHook for StepHook {
    type E = SumTargetEnv;

    fn hook<S: Clone + Send + Sync + 'static>(
        &mut self,
        _core: &mut R2lSamplerCore<Self::E, S>,
    ) -> SamplerHookResult {
        self.collect = !self.collect;
        if self.collect {
            SamplerHookResult::Bound(RolloutMode::StepBound {
                n_steps: STEPS_PER_ROLLOUT,
            })
        } else {
            SamplerHookResult::Stop
        }
    }
}

struct LearningHook {
    epoch: usize,
}

impl RecurrentPPOHook<Module> for LearningHook {
    fn before_learning_hook<
        B: TrajectoryBatch<
                <Module as OnPolicyLearningModule>::InferenceTensor,
                State = <Module as OnPolicyLearningModule>::InferenceState,
            >,
    >(
        &mut self,
        _params: &mut RecurrentPPOParams,
        _module: &mut Module,
        _batches: &[B],
        advantages: &mut Advantages,
        _returns: &mut Returns,
    ) -> anyhow::Result<HookResult> {
        self.epoch = 0;
        advantages.normalize();
        Ok(HookResult::Continue)
    }

    fn rollout_hook<
        B: TrajectoryBatch<
                <Module as OnPolicyLearningModule>::InferenceTensor,
                State = <Module as OnPolicyLearningModule>::InferenceState,
            >,
    >(
        &mut self,
        _params: &mut RecurrentPPOParams,
        _module: &mut Module,
        _batches: &[B],
    ) -> anyhow::Result<HookResult> {
        self.epoch += 1;
        Ok(if self.epoch >= PPO_EPOCHS {
            HookResult::Break
        } else {
            HookResult::Continue
        })
    }

    fn batch_hook(
        &mut self,
        _params: &mut RecurrentPPOParams,
        _module: &mut Module,
        losses: &mut <Module as LearningModule>::Losses,
        _data: &RecurrentPPOBatchData<<Module as OnPolicyLearningModule>::LearningTensor>,
    ) -> anyhow::Result<HookResult> {
        losses.set_vf_coeff(Some(0.5));
        Ok(HookResult::Continue)
    }
}

fn evaluate(
    policy: &RecurrentDiagGaussianDistribution<InferenceBackend>,
    carry_state: bool,
) -> f32 {
    InferenceBackend::seed(&Default::default(), 91);
    let mut total_reward = 0.;
    let mut total_steps = 0;
    for seed in 100..132 {
        let mut env = SumTargetEnv::new(EPISODE_STEPS);
        let mut observation = env.reset(seed).unwrap();
        let mut state = None;
        loop {
            let observation_tensor = Tensor::<InferenceBackend, 1>::convert(&observation);
            let (action, next_state) = policy.action(observation_tensor, state).unwrap();
            let snapshot = env.step(TensorData::convert(&action)).unwrap();
            total_reward += snapshot.reward;
            total_steps += 1;
            let done = snapshot.done();
            observation = snapshot.state;
            state = carry_state.then_some(next_state);
            if done {
                break;
            }
        }
    }
    total_reward / total_steps as f32
}

#[test]
fn recurrent_ppo_improves_sum_target_reward() {
    set_seed(0);
    TrainBackend::seed(&Default::default(), 0);
    let policy = Policy::build(&[3, 32, 1]);
    let module = PolicyValueModule::split(
        policy,
        &[3, 32, 1],
        r2l_core::models::ActivationFunction::Tanh,
        AdamWConfig::new(),
        3e-4,
        AdamWConfig::new(),
        1e-3,
    );
    let agent = RecurrentPPO {
        params: RecurrentPPOParams {
            sequence_length: EPISODE_STEPS,
            gamma: 0.95,
            lambda: 0.95,
            ..Default::default()
        },
        lm: module,
        hooks: LearningHook { epoch: 0 },
    };
    let env_builder = EnvBuilderType::homogenous(|| Ok(SumTargetEnv::new(EPISODE_STEPS)), N_ENVS);
    let sampler = R2lSampler::build(
        env_builder,
        StepHook { collect: false },
        SamplerExecutionMode::Vec,
    );
    let mut runtime = OnPolicyRuntime {
        agent,
        sampler,
        adapter: DefaultAdapter,
    };

    let reward_before = evaluate(&runtime.actor(), true);
    for _ in 0..TRAINING_ROLLOUTS {
        runtime.collect();
        runtime.learn().unwrap();
    }
    let trained_actor = runtime.actor();
    let reward_after = evaluate(&trained_actor, true);
    let reward_without_memory = evaluate(&trained_actor, false);
    runtime.shutdown();

    println!("reward before={reward_before} and after={reward_after}");

    assert!(
        reward_after > reward_before + 0.25,
        "expected recurrent PPO to improve reward, before={reward_before}, after={reward_after}"
    );
    assert!(
        reward_after > reward_without_memory + 0.05,
        "expected carried recurrent state to help, stateful={reward_after}, reset={reward_without_memory}"
    );
}
