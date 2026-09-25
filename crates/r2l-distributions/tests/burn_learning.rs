use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    module::{AutodiffModule, Module, ModuleVisitor, Param, ParamId},
    optim::AdamWConfig,
    prelude::Backend,
    record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    tensor::Tensor,
};
use r2l_core::{
    models::{ActivationFunction, Actor, Learner},
    tensor::R2lTensor,
};
use r2l_distributions::{
    Categorical, Composite, DiagGaussian, DistributionKind, MultiBernoulli, MultiCategorical,
    Network, OnPolicyLearner2, Policy2, ValueFunction2,
    learning_modules::burn_lm::{
        BurnDistributionKind, BurnPolicy, NetworkKind, PolicyValueLearner, PolicyValueLosses,
    },
    networks::mlp::Mlp,
};

type B = Autodiff<NdArray>;

fn mlp(outputs: usize) -> Mlp<B> {
    Mlp::build(&[2, outputs], ActivationFunction::Tanh)
}

fn network(outputs: usize) -> NetworkKind<B> {
    NetworkKind::Mlp(mlp(outputs))
}

fn build_learner<P: BurnPolicy<B>>(policy: P, split: bool) -> PolicyValueLearner<B, P> {
    let optimizer = AdamWConfig::new().with_weight_decay(0.0);
    if split {
        PolicyValueLearner::split_with_network(
            policy,
            network(1),
            &optimizer,
            0.01,
            &optimizer,
            0.01,
        )
    } else {
        PolicyValueLearner::joint_with_network(policy, network(1), &optimizer, 0.01)
    }
}

fn update<P: BurnPolicy<B>>(learner: &mut PolicyValueLearner<B, P>, actions: Tensor<B, 2>) {
    let observations = Tensor::zeros([2, 2], &NdArrayDevice::Cpu);
    let log_probs = learner
        .policy()
        .log_probs(observations.clone(), actions)
        .unwrap();
    let entropy_loss = learner
        .policy()
        .entropy(observations.clone())
        .unwrap()
        .mul_scalar(-0.01);
    let values = learner.values(observations).unwrap();
    let returns = learner.tensor_from_slice(&[1.0, 1.0]).unwrap();
    assert_eq!(log_probs.dims(), [2, 1]);
    assert_eq!(values.dims(), [2, 1]);
    assert_eq!(returns.dims(), [2, 1]);
    let policy_loss = R2lTensor::mean(&log_probs.neg()).unwrap();
    let value_loss = R2lTensor::mean(&(values - returns).powf_scalar(2.0)).unwrap();
    let mut losses = PolicyValueLosses::new(policy_loss, value_loss);
    losses.add_entropy_loss(entropy_loss);
    losses.set_vf_coeff(Some(0.5));
    learner.update(losses).unwrap();
}

fn restore<BackendType: Backend, M: Module<BackendType>>(target: M, source: M) -> M {
    let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
    let bytes = Recorder::<BackendType>::record(&recorder, source.into_record(), ()).unwrap();
    let record =
        Recorder::<BackendType>::load(&recorder, bytes, &BackendType::Device::default()).unwrap();
    target.load_record(record)
}

#[derive(Default)]
struct ParameterIds(Vec<ParamId>);

impl<BackendType: Backend> ModuleVisitor<BackendType> for ParameterIds {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<BackendType, D>>) {
        self.0.push(param.id);
    }
}

fn parameter_ids<BackendType: Backend>(module: &impl Module<BackendType>) -> Vec<ParamId> {
    let mut visitor = ParameterIds::default();
    module.visit(&mut visitor);
    visitor.0
}

#[test]
fn categorical_joint_and_split_learners_update_policy_and_value() {
    for split in [false, true] {
        let policy = Categorical::new(mlp(2)).unwrap();
        let mut learner = build_learner(policy, split);
        let observations = Tensor::zeros([2, 2], &NdArrayDevice::Cpu);
        let actions = Tensor::zeros([2, 1], &NdArrayDevice::Cpu);
        let before = learner
            .policy()
            .log_probs(observations.clone(), actions.clone())
            .unwrap()
            .to_vec()
            .unwrap();
        let value_before = learner
            .values(observations.clone())
            .unwrap()
            .to_vec()
            .unwrap();
        let snapshot = learner.inference_policy();
        for _ in 0..2 {
            update(&mut learner, actions.clone());
        }
        let after = learner
            .policy()
            .log_probs(observations.clone(), actions)
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(after[0] > before[0]);
        assert_ne!(
            value_before,
            learner.values(observations).unwrap().to_vec().unwrap()
        );
        let inference = learner.inference_policy();
        let input = Tensor::<NdArray, 2>::zeros([2, 2], &NdArrayDevice::Cpu);
        let actions = Tensor::<NdArray, 2>::zeros([2, 1], &NdArrayDevice::Cpu);
        let inference_log_probs = inference.log_probs(input.clone(), actions.clone()).unwrap();
        assert!(!inference_log_probs.is_require_grad());
        assert_eq!(after, inference_log_probs.to_vec().unwrap());
        assert_eq!(
            before,
            snapshot
                .log_probs(input.clone(), actions.clone())
                .unwrap()
                .to_vec()
                .unwrap()
        );
        let restored = restore::<NdArray, _>(snapshot, inference);
        assert_eq!(
            after,
            restored
                .log_probs(input, actions)
                .unwrap()
                .to_vec()
                .unwrap()
        );
    }
}

#[test]
fn gaussian_updates_mean_and_log_std_and_preserves_parameter_ids() {
    for split in [false, true] {
        let log_std = Param::from_tensor(Tensor::<B, 2>::zeros([1, 1], &NdArrayDevice::Cpu));
        let std_id = log_std.id;
        let policy = DiagGaussian::new(mlp(1), log_std).unwrap();
        let mut learner = build_learner(policy, split);
        let observations = Tensor::zeros([1, 2], &NdArrayDevice::Cpu);
        let mean_before = learner
            .policy()
            .mode_action(observations.clone())
            .unwrap()
            .to_vec()
            .unwrap();
        let snapshot = learner.inference_policy();
        let trainable_template = learner.policy().clone();
        let ids = parameter_ids::<B>(learner.policy());
        assert!(ids.contains(&std_id));
        assert_eq!(learner.policy().std().unwrap(), Some(1.0));
        for _ in 0..2 {
            update(&mut learner, Tensor::full([2, 1], 2.0, &NdArrayDevice::Cpu));
        }
        let trained_std = learner.policy().std().unwrap().unwrap();
        assert!(trained_std > 1.0);
        assert_ne!(
            mean_before,
            learner
                .policy()
                .mode_action(observations)
                .unwrap()
                .to_vec()
                .unwrap()
        );
        assert_eq!(ids, parameter_ids::<B>(learner.policy()));
        let inference = learner.inference_policy();
        assert_eq!(inference.std().unwrap(), Some(trained_std));
        assert_eq!(snapshot.std().unwrap(), Some(1.0));
        let trainable =
            DiagGaussian::<Mlp<B>, Param<Tensor<B, 2>>>::from_inner(learner.inference_policy());
        assert_eq!(ids, parameter_ids::<B>(&trainable));
        let mut from_inference = build_learner(trainable, split);
        update(
            &mut from_inference,
            Tensor::full([2, 1], 2.0, &NdArrayDevice::Cpu),
        );
        assert!(from_inference.policy().std().unwrap().unwrap() > trained_std);
        let restored = restore::<NdArray, _>(snapshot, inference);
        assert_eq!(restored.std().unwrap(), Some(trained_std));
        let restored = restore::<B, _>(trainable_template, learner.policy().clone());
        let mut resumed = build_learner(restored, split);
        update(&mut resumed, Tensor::full([2, 1], 2.0, &NdArrayDevice::Cpu));
        assert!(resumed.policy().std().unwrap().unwrap() > trained_std);
        assert_eq!(ids, parameter_ids::<B>(resumed.policy()));
    }
}

#[test]
fn nested_composite_updates_and_round_trips_all_distribution_variants() {
    let children = vec![
        DistributionKind::DiagGaussian(
            DiagGaussian::new(
                network(1),
                Param::from_tensor(Tensor::zeros([1, 1], &NdArrayDevice::Cpu)),
            )
            .unwrap(),
        ),
        DistributionKind::MultiBernoulli(MultiBernoulli::new(network(1)).unwrap()),
        DistributionKind::MultiCategorical(MultiCategorical::new(network(5), vec![2, 3]).unwrap()),
    ];
    let policy: BurnDistributionKind<B> = DistributionKind::Composite(
        Composite::new(vec![
            DistributionKind::Categorical(Categorical::new(network(2)).unwrap()),
            DistributionKind::Composite(Composite::new(children).unwrap()),
        ])
        .unwrap(),
    );
    assert_eq!(policy.action_shape().dims(), [5]);
    assert_eq!(Module::num_params(&policy), 28);
    let mut learner: PolicyValueLearner<B> = build_learner(policy, false);
    let observations = Tensor::<B, 2>::zeros([2, 2], &NdArrayDevice::Cpu);
    let actions = Tensor::from_data([[0.0, 2.0, 1.0, 0.0, 1.0]; 2], &NdArrayDevice::Cpu);
    let before = learner
        .policy()
        .log_probs(observations.clone(), actions.clone())
        .unwrap()
        .to_vec()
        .unwrap();
    let snapshot = learner.inference_policy();
    let trainable_template = learner.policy().clone();
    update(&mut learner, actions.clone());
    let after = learner
        .policy()
        .log_probs(observations, actions.clone())
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(after[0] > before[0]);
    let inference = learner.inference_policy();
    let restored = restore::<NdArray, _>(snapshot, inference);
    assert_eq!(restored.action_shape().dims(), [5]);
    assert_eq!(
        after,
        restored
            .log_probs(Tensor::zeros([2, 2], &NdArrayDevice::Cpu), actions.inner())
            .unwrap()
            .to_vec()
            .unwrap()
    );
    let mode = restored
        .mode_action(Tensor::zeros([1, 2], &NdArrayDevice::Cpu))
        .unwrap();
    assert_eq!(mode.dims(), [1, 5]);
    let restored = restore::<B, _>(trainable_template, learner.policy().clone());
    let mut resumed: PolicyValueLearner<B> = build_learner(restored, false);
    let actions = Tensor::from_data([[0.0, 2.0, 1.0, 0.0, 1.0]; 2], &NdArrayDevice::Cpu);
    update(&mut resumed, actions.clone());
    let resumed_log_probs = resumed
        .policy()
        .log_probs(Tensor::zeros([2, 2], &NdArrayDevice::Cpu), actions)
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(resumed_log_probs[0] > after[0]);
}

#[test]
fn network_and_value_function_reject_invalid_batches() {
    let network = network(1);
    assert!(
        network
            .forward(Tensor::zeros([1, 3], &NdArrayDevice::Cpu))
            .is_err()
    );
    assert!(
        network
            .forward(Tensor::zeros([0, 2], &NdArrayDevice::Cpu))
            .is_err()
    );
    let learner = build_learner(Categorical::new(mlp(2)).unwrap(), false);
    assert!(
        learner
            .values(Tensor::zeros([1, 3], &NdArrayDevice::Cpu))
            .is_err()
    );
    assert!(
        learner
            .values(Tensor::zeros([0, 2], &NdArrayDevice::Cpu))
            .is_err()
    );
    let inference = Tensor::<NdArray, 2>::from_data([[1.0, 2.0]], &NdArrayDevice::Cpu);
    let lifted = <PolicyValueLearner<B> as OnPolicyLearner2>::lifter(&inference);
    assert_eq!(lifted.dims(), [1, 2]);
    assert_eq!(lifted.device(), inference.device());
    assert_eq!(lifted.to_vec().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn raw_gaussian_preserves_externally_managed_parameter_gradients() {
    let log_std = Tensor::<B, 2>::zeros([1, 1], &NdArrayDevice::Cpu).require_grad();
    let policy: DiagGaussian<Mlp<B>> = DiagGaussian::new(mlp(1), log_std.clone()).unwrap();
    let entropy = policy
        .entropy(Tensor::zeros([2, 2], &NdArrayDevice::Cpu))
        .unwrap();
    let grads = entropy.backward();
    assert_eq!(log_std.grad(&grads).unwrap().to_vec().unwrap(), vec![1.0]);
}
