use burn::{
    module::{Module, ModuleVisitor, Param, ParamId},
    optim::AdamWConfig,
    store::ModuleRecord,
    tensor::{Device, Tensor},
};
use r2l_core::{
    models::{ActivationFunction, Actor, Learner},
    tensor::R2lTensor,
};
use r2l_distributions::{
    Categorical, Composite, DiagGaussian, DistributionKind, MultiBernoulli, MultiCategorical,
    Network, OnPolicyLearner, Policy, ValueFunction,
    learning_modules::burn_lm::{
        BurnDistributionKind, BurnPolicy, NetworkKind, PolicyValueLearner, PolicyValueLosses,
    },
    networks::burn::mlp::Mlp,
};

fn mlp(outputs: usize) -> Mlp {
    Mlp::build(&[2, outputs], ActivationFunction::Tanh).train()
}

fn network(outputs: usize) -> NetworkKind {
    NetworkKind::Mlp(mlp(outputs))
}

fn build_learner<P: BurnPolicy>(policy: P, split: bool) -> PolicyValueLearner<P> {
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

fn update<P: BurnPolicy>(learner: &mut PolicyValueLearner<P>, actions: Tensor<2>) {
    let observations = Tensor::zeros([2, 2], &Device::flex().autodiff());
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
    losses.add_entropy_loss(&entropy_loss).unwrap();
    losses.set_vf_coeff(0.5);
    learner.update(losses).unwrap();
}

fn restore<M: Module>(target: M, source: M) -> M {
    let bytes = source.into_record().into_bytes().unwrap();
    let record = ModuleRecord::from_bytes(bytes).unwrap();
    target.load_record(record)
}

#[derive(Default)]
struct ParameterIds(Vec<ParamId>);

impl ModuleVisitor for ParameterIds {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.0.push(param.id);
    }
}

fn parameter_ids(module: &impl Module) -> Vec<ParamId> {
    let mut visitor = ParameterIds::default();
    module.visit(&mut visitor);
    visitor.0
}

#[test]
fn categorical_joint_and_split_learners_update_policy_and_value() {
    for split in [false, true] {
        let policy = Categorical::new(mlp(2)).unwrap();
        let mut learner = build_learner(policy, split);
        let observations = Tensor::zeros([2, 2], &Device::flex().autodiff());
        let actions = Tensor::zeros([2, 1], &Device::flex().autodiff());
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
        let input = Tensor::<2>::zeros([2, 2], &Device::flex());
        let actions = Tensor::<2>::zeros([2, 1], &Device::flex());
        let inference_log_probs = inference.log_probs(input.clone(), actions.clone()).unwrap();
        assert!(!inference_log_probs.is_autodiff());
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
        let restored = restore(snapshot, inference);
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
        let log_std = Param::from_tensor(Tensor::<2>::zeros([1, 1], &Device::flex().autodiff()));
        let std_id = log_std.id;
        let policy = DiagGaussian::new(mlp(1), log_std).unwrap();
        let mut learner = build_learner(policy, split);
        let observations = Tensor::zeros([1, 2], &Device::flex().autodiff());
        let mean_before = learner
            .policy()
            .mode_action(observations.clone())
            .unwrap()
            .to_vec()
            .unwrap();
        let snapshot = learner.inference_policy();
        let trainable_template = learner.policy().clone();
        let ids = parameter_ids(learner.policy());
        assert!(ids.contains(&std_id));
        assert_eq!(learner.policy().std().unwrap(), Some(1.0));
        for _ in 0..2 {
            update(
                &mut learner,
                Tensor::full([2, 1], 2.0, &Device::flex().autodiff()),
            );
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
        assert_eq!(ids, parameter_ids(learner.policy()));
        let inference = learner.inference_policy();
        assert_eq!(inference.std().unwrap(), Some(trained_std));
        assert_eq!(snapshot.std().unwrap(), Some(1.0));
        let trainable = learner.inference_policy().train();
        assert_eq!(ids, parameter_ids(&trainable));
        let mut from_inference = build_learner(trainable, split);
        update(
            &mut from_inference,
            Tensor::full([2, 1], 2.0, &Device::flex().autodiff()),
        );
        assert!(from_inference.policy().std().unwrap().unwrap() > trained_std);
        let restored = restore(snapshot, inference);
        assert_eq!(restored.std().unwrap(), Some(trained_std));
        let restored = restore(trainable_template, learner.policy().clone());
        let mut resumed = build_learner(restored, split);
        update(
            &mut resumed,
            Tensor::full([2, 1], 2.0, &Device::flex().autodiff()),
        );
        assert!(resumed.policy().std().unwrap().unwrap() > trained_std);
        assert_eq!(ids, parameter_ids(resumed.policy()));
    }
}

#[test]
fn nested_composite_updates_and_round_trips_all_distribution_variants() {
    let children = vec![
        DistributionKind::DiagGaussian(
            DiagGaussian::new(
                network(1),
                Param::from_tensor(Tensor::zeros([1, 1], &Device::flex().autodiff())),
            )
            .unwrap(),
        ),
        DistributionKind::MultiBernoulli(MultiBernoulli::new(network(1)).unwrap()),
        DistributionKind::MultiCategorical(MultiCategorical::new(network(5), vec![2, 3]).unwrap()),
    ];
    let policy: BurnDistributionKind = DistributionKind::Composite(
        Composite::new(vec![
            DistributionKind::Categorical(Categorical::new(network(2)).unwrap()),
            DistributionKind::Composite(Composite::new(children).unwrap()),
        ])
        .unwrap(),
    );
    assert_eq!(policy.action_shape().dims(), [5]);
    assert_eq!(Module::num_params(&policy), 28);
    let mut learner: PolicyValueLearner = build_learner(policy, false);
    let observations = Tensor::<2>::zeros([2, 2], &Device::flex().autodiff());
    let actions = Tensor::from_data([[0.0, 2.0, 1.0, 0.0, 1.0]; 2], &Device::flex().autodiff());
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
    let restored = restore(snapshot, inference);
    assert_eq!(restored.action_shape().dims(), [5]);
    assert_eq!(
        after,
        restored
            .log_probs(Tensor::zeros([2, 2], &Device::flex()), actions.inner())
            .unwrap()
            .to_vec()
            .unwrap()
    );
    let mode = restored
        .mode_action(Tensor::zeros([1, 2], &Device::flex()))
        .unwrap();
    assert_eq!(mode.dims(), [1, 5]);
    let restored = restore(trainable_template, learner.policy().clone());
    let mut resumed: PolicyValueLearner = build_learner(restored, false);
    let actions = Tensor::from_data([[0.0, 2.0, 1.0, 0.0, 1.0]; 2], &Device::flex().autodiff());
    update(&mut resumed, actions.clone());
    let resumed_log_probs = resumed
        .policy()
        .log_probs(Tensor::zeros([2, 2], &Device::flex().autodiff()), actions)
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
            .forward(Tensor::zeros([1, 3], &Device::flex().autodiff()))
            .is_err()
    );
    assert!(
        network
            .forward(Tensor::zeros([0, 2], &Device::flex().autodiff()))
            .is_err()
    );
    let learner = build_learner(Categorical::new(mlp(2)).unwrap(), false);
    assert!(
        learner
            .values(Tensor::zeros([1, 3], &Device::flex().autodiff()))
            .is_err()
    );
    assert!(
        learner
            .values(Tensor::zeros([0, 2], &Device::flex().autodiff()))
            .is_err()
    );
    let inference = Tensor::<2>::from_data([[1.0, 2.0]], &Device::flex());
    let prepared = <PolicyValueLearner as OnPolicyLearner>::prepare_learning_tensor(&inference);
    assert_eq!(prepared.dims(), [1, 2]);
    assert!(prepared.is_autodiff());
    assert_eq!(prepared.clone().inner().device(), inference.device());
    assert_eq!(prepared.to_vec().unwrap(), vec![1.0, 2.0]);

    let input = inference.autodiff().require_grad();
    let rollout = input.clone().mul_scalar(2.0);
    let prepared =
        <PolicyValueLearner as OnPolicyLearner>::prepare_learning_tensor(&rollout).require_grad();
    let gradients = prepared.powf_scalar(2.0).sum().backward();
    assert!(input.grad(&gradients).is_none());
}

#[test]
fn raw_gaussian_preserves_externally_managed_parameter_gradients() {
    let log_std = Tensor::<2>::zeros([1, 1], &Device::flex().autodiff()).require_grad();
    let policy: DiagGaussian<Mlp> = DiagGaussian::new(mlp(1), log_std.clone()).unwrap();
    let inference = policy.for_inference();
    let log_probs = inference
        .log_probs(
            Tensor::zeros([2, 2], &Device::flex()),
            Tensor::zeros([2, 1], &Device::flex()),
        )
        .unwrap();
    assert!(!log_probs.is_require_grad());
    let entropy = policy
        .entropy(Tensor::zeros([2, 2], &Device::flex().autodiff()))
        .unwrap();
    let grads = entropy.backward();
    assert_eq!(log_std.grad(&grads).unwrap().to_vec().unwrap(), vec![1.0]);
}
