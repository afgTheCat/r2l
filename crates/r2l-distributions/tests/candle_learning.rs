use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{Init, ParamsAdamW, VarBuilder, VarMap};
use r2l_core::models::{ActivationFunction, Actor, Learner};
use r2l_distributions::{
    Categorical, Composite, DiagGaussian, DistributionKind, MultiBernoulli, MultiCategorical,
    Network, OnPolicyLearner, Policy, ValueFunction,
    learning_modules::candle_lm::{PolicyValueLearner, PolicyValueLosses, PolicyValueOptimizer},
    networks::candle_mlp::Mlp,
};

fn builder(vm: &VarMap) -> VarBuilder<'_> {
    VarBuilder::from_varmap(vm, DType::F32, &Device::Cpu)
}

fn network(vb: &VarBuilder<'_>, outputs: usize) -> Mlp {
    Mlp::build(&[2, outputs], ActivationFunction::Tanh, vb).unwrap()
}

fn zeros(rows: usize, columns: usize) -> Tensor {
    Tensor::zeros((rows, columns), DType::F32, &Device::Cpu).unwrap()
}

fn data(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

fn build_learner<P: Policy<Tensor = Tensor> + Clone>(
    policy: P,
    policy_vm: &VarMap,
    split: bool,
) -> PolicyValueLearner<P> {
    let value_vm = if split {
        VarMap::new()
    } else {
        policy_vm.clone()
    };
    let value = network(&builder(&value_vm).pp("value"), 1);
    for var in policy_vm.all_vars().into_iter().chain(value_vm.all_vars()) {
        var.set(&var.zeros_like().unwrap()).unwrap();
    }
    let params = ParamsAdamW {
        lr: 0.01,
        weight_decay: 0.0,
        ..Default::default()
    };
    let optimizer = if split {
        PolicyValueOptimizer::split(policy_vm, &value_vm, params.clone(), params, None, None)
    } else {
        PolicyValueOptimizer::joint(policy_vm, params, None)
    }
    .unwrap();
    PolicyValueLearner::new(policy, value, optimizer, Device::Cpu).unwrap()
}

fn update<P: Policy<Tensor = Tensor> + Clone>(
    learner: &mut PolicyValueLearner<P>,
    actions: &Tensor,
) {
    let observations = zeros(2, 2);
    let log_probs = learner
        .policy()
        .log_probs(observations.clone(), actions.clone())
        .unwrap();
    let entropy_loss = (learner.policy().entropy(observations.clone()).unwrap() * -0.01).unwrap();
    let values = learner.values(observations).unwrap();
    let returns = learner.tensor_from_slice(&[1.0, 1.0]).unwrap();
    assert_eq!(log_probs.dims(), [2, 1]);
    assert_eq!(values.dims(), [2, 1]);
    assert_eq!(returns.dims(), [2, 1]);
    let policy_loss = log_probs.neg().unwrap().mean_all().unwrap();
    let value_loss = (values - returns)
        .unwrap()
        .sqr()
        .unwrap()
        .mean_all()
        .unwrap();
    let mut losses = PolicyValueLosses::new(policy_loss, value_loss);
    losses.add_entropy_loss(&entropy_loss).unwrap();
    losses.set_vf_coeff(Some(0.5));
    learner.update(losses).unwrap();
}

#[test]
fn categorical_joint_and_split_update_policy_and_value() {
    for split in [false, true] {
        let vm = VarMap::new();
        let policy = Categorical::new(network(&builder(&vm).pp("policy"), 2)).unwrap();
        let mut learner = build_learner(policy, &vm, split);
        let observations = zeros(2, 2);
        let actions = zeros(2, 1);
        let before = data(
            &learner
                .policy()
                .log_probs(observations.clone(), actions.clone())
                .unwrap(),
        );
        let inference = learner.inference_policy();
        let old_log_probs = inference
            .log_probs(observations.clone(), actions.clone())
            .unwrap();
        for _ in 0..2 {
            update(&mut learner, &actions);
        }
        let after = data(
            &learner
                .policy()
                .log_probs(observations.clone(), actions.clone())
                .unwrap(),
        );
        assert!(after[0] > before[0]);
        assert!(data(&learner.values(observations.clone()).unwrap())[0] > 0.0);
        // Clones read live weights, while previously collected log probabilities stay fixed.
        assert_eq!(before, data(&old_log_probs));
        assert_eq!(
            after,
            data(&inference.log_probs(observations, actions).unwrap())
        );
        learner.set_learning_rate(0.002);
        assert_eq!(learner.policy_learning_rate(), 0.002);
    }
}

#[test]
fn gaussian_updates_mean_and_log_std_and_detaches_rollout_outputs() {
    for split in [false, true] {
        let vm = VarMap::new();
        let vb = builder(&vm).pp("policy");
        let log_std = vb
            .get_with_hints((1, 1), "log_std", Init::Const(0.0))
            .unwrap();
        let policy = DiagGaussian::new(network(&vb.pp("mean"), 1), log_std.clone()).unwrap();
        let mut learner = build_learner(policy, &vm, split);
        let actions = Tensor::full(2.0f32, (2, 1), &Device::Cpu).unwrap();
        for _ in 0..2 {
            update(&mut learner, &actions);
        }
        assert!(data(&learner.policy().mode_action(zeros(1, 2)).unwrap())[0] > 0.0);
        assert!(learner.policy().std().unwrap().unwrap() > 1.0);
        assert!(data(&log_std)[0] > 0.0);
        let input = Var::from_tensor(&zeros(1, 2)).unwrap();
        let inference = learner.inference_policy();
        let outputs = [
            inference.action(input.as_tensor().clone()).unwrap(),
            inference.mode_action(input.as_tensor().clone()).unwrap(),
            inference
                .log_probs(input.as_tensor().clone(), zeros(1, 1))
                .unwrap(),
            inference.entropy(input.as_tensor().clone()).unwrap(),
        ];
        for output in outputs {
            let grads = output.backward().unwrap();
            assert!(grads.get(&input).is_none());
            for var in vm.all_vars() {
                assert!(grads.get(&var).is_none());
            }
        }
    }
}

#[test]
fn nested_composite_trains_every_distribution_variant() {
    for split in [false, true] {
        let vm = VarMap::new();
        let vb = builder(&vm).pp("policy");
        let children = vec![
            DistributionKind::DiagGaussian(
                DiagGaussian::new(
                    network(&vb.pp("gaussian"), 1),
                    vb.get_with_hints((1, 1), "log_std", Init::Const(0.0))
                        .unwrap(),
                )
                .unwrap(),
            ),
            DistributionKind::MultiBernoulli(
                MultiBernoulli::new(network(&vb.pp("bernoulli"), 1)).unwrap(),
            ),
            DistributionKind::MultiCategorical(
                MultiCategorical::new(network(&vb.pp("multi"), 5), vec![2, 3]).unwrap(),
            ),
        ];
        let policy = DistributionKind::Composite(
            Composite::new(vec![
                DistributionKind::Categorical(
                    Categorical::new(network(&vb.pp("categorical"), 2)).unwrap(),
                ),
                DistributionKind::Composite(Composite::new(children).unwrap()),
            ])
            .unwrap(),
        );
        let mut learner: PolicyValueLearner = build_learner(policy, &vm, split);
        let actions = Tensor::new(&[[0.0f32, 2.0, 1.0, 0.0, 1.0]; 2], &Device::Cpu).unwrap();
        let before = data(
            &learner
                .policy()
                .log_probs(zeros(2, 2), actions.clone())
                .unwrap(),
        );
        update(&mut learner, &actions);
        let after = data(&learner.policy().log_probs(zeros(2, 2), actions).unwrap());
        assert!(after[0] > before[0]);
        for (name, var) in vm.data().lock().unwrap().iter() {
            if name.starts_with("policy.") && !name.ends_with("weight") {
                assert!(
                    data(var).iter().any(|v| *v != 0.0),
                    "unchanged parameter: {name}"
                );
            }
        }
        let inference = learner.inference_policy();
        assert_eq!(inference.action(zeros(1, 2)).unwrap().dims(), [1, 5]);
        assert_eq!(inference.mode_action(zeros(1, 2)).unwrap().dims(), [1, 5]);
    }
}

fn scalar_parameter(vm: &VarMap, name: &str) -> Tensor {
    builder(vm)
        .get_with_hints((), name, Init::Const(1.0))
        .unwrap()
}

#[test]
fn hidden_layers_preserve_batch_shape_and_parameter_gradients() {
    let vm = VarMap::new();
    let net = Mlp::build(&[2, 3, 1], ActivationFunction::Tanh, &builder(&vm)).unwrap();
    for var in vm.all_vars() {
        var.set(&Tensor::full(0.1f32, var.shape(), var.device()).unwrap())
            .unwrap();
    }
    let observations = Tensor::new(&[[1.0f32, 2.0], [2.0, 3.0]], &Device::Cpu).unwrap();
    let output = net.forward(observations).unwrap();
    assert_eq!(output.dims(), [2, 1]);
    let expected = [0.3 * 0.4f32.tanh() + 0.1, 0.3 * 0.6f32.tanh() + 0.1];
    for (actual, expected) in data(&output).into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6);
    }
    let grads = output.sum_all().unwrap().backward().unwrap();
    for var in vm.all_vars() {
        assert!(
            data(grads.get(&var).unwrap())
                .iter()
                .all(|gradient| *gradient > 0.0)
        );
    }
}

#[test]
fn joint_deduplicates_shared_parameters_and_split_backprops_before_stepping() {
    let params = ParamsAdamW {
        lr: 1.0,
        beta1: 0.0,
        beta2: 0.0,
        eps: 1.0,
        weight_decay: 0.0,
    };
    let vm = VarMap::new();
    let shared = scalar_parameter(&vm, "shared");
    let var = vm.all_vars().pop().unwrap();
    vm.data().lock().unwrap().insert("alias".into(), var);
    let mut joint = PolicyValueOptimizer::joint(&vm, params.clone(), None).unwrap();
    joint
        .update(PolicyValueLosses::new(
            (&shared * 2.0).unwrap(),
            (&shared * 3.0).unwrap(),
        ))
        .unwrap();
    assert!((data(&shared)[0] - (1.0 - 5.0 / 6.0)).abs() < 1e-6);

    let policy_vm = VarMap::new();
    let value_vm = VarMap::new();
    let policy = scalar_parameter(&policy_vm, "policy");
    let value = scalar_parameter(&value_vm, "value");
    let mut split =
        PolicyValueOptimizer::split(&policy_vm, &value_vm, params.clone(), params, None, None)
            .unwrap();
    split
        .update(PolicyValueLosses::new(
            policy.sqr().unwrap(),
            (&policy * &value).unwrap(),
        ))
        .unwrap();
    assert!((data(&value)[0] - 0.5).abs() < 1e-6);
}

#[test]
fn optimizer_respects_loss_coefficients_clipping_and_learning_rates() {
    for split in [false, true] {
        let policy_vm = VarMap::new();
        let value_vm = if split {
            VarMap::new()
        } else {
            policy_vm.clone()
        };
        let policy = scalar_parameter(&policy_vm, "policy");
        let value = scalar_parameter(&value_vm, "value");
        // Nonzero epsilon makes clipping observable even on Adam's first step.
        let params = ParamsAdamW {
            lr: 1.0,
            beta1: 0.0,
            beta2: 0.0,
            eps: 1.0,
            weight_decay: 0.0,
        };
        let mut optimizer = if split {
            PolicyValueOptimizer::split(
                &policy_vm,
                &value_vm,
                params.clone(),
                params,
                Some(1.0),
                Some(1.0),
            )
        } else {
            PolicyValueOptimizer::joint(&policy_vm, params, Some(1.0))
        }
        .unwrap();
        let mut losses = PolicyValueLosses::new((&policy * 2.0).unwrap(), (&value * 8.0).unwrap());
        losses.add_entropy_loss(&policy).unwrap(); // policy gradient = 3
        losses.set_vf_coeff(Some(0.5)); // value gradient = 4
        optimizer.update(losses).unwrap();
        let (policy_grad, value_grad) = if split { (1.0, 1.0) } else { (0.6, 0.8) };
        assert!((data(&policy)[0] - (1.0 - policy_grad / (policy_grad + 1.0))).abs() < 1e-6);
        assert!((data(&value)[0] - (1.0 - value_grad / (value_grad + 1.0))).abs() < 1e-6);
        let before_policy = data(&policy);
        let before_value = data(&value);
        optimizer.set_learning_rate(0.0);
        optimizer
            .update(PolicyValueLosses::new(
                policy.sqr().unwrap(),
                value.sqr().unwrap(),
            ))
            .unwrap();
        assert_eq!(data(&policy), before_policy);
        assert_eq!(data(&value), before_value);
    }
}

#[test]
fn rejects_invalid_shapes_and_overlapping_split_variables() {
    let vm = VarMap::new();
    let vb = builder(&vm);
    for widths in [&[][..], &[2], &[0, 1], &[2, 0]] {
        assert!(Mlp::build(widths, ActivationFunction::Tanh, &vb).is_err());
    }
    let net = network(&vb.pp("net"), 1);
    for shape in [vec![2], vec![1, 3], vec![0, 2], vec![1, 1, 2]] {
        assert!(
            net.forward(Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap())
                .is_err()
        );
    }
    assert!(
        PolicyValueOptimizer::split(
            &vm,
            &vm.clone(),
            ParamsAdamW::default(),
            ParamsAdamW::default(),
            None,
            None
        )
        .is_err()
    );
    for norm in [-1.0, f32::NAN, f32::INFINITY] {
        assert!(PolicyValueOptimizer::joint(&vm, ParamsAdamW::default(), Some(norm)).is_err());
    }
    let policy = Categorical::new(network(&vb.pp("policy"), 2)).unwrap();
    let invalid_value = network(&vb.pp("value"), 2);
    let optimizer = PolicyValueOptimizer::joint(&vm, ParamsAdamW::default(), None).unwrap();
    assert!(PolicyValueLearner::new(policy, invalid_value, optimizer, Device::Cpu).is_err());
    let input = Var::from_tensor(&Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap()).unwrap();
    let lifted = <PolicyValueLearner as OnPolicyLearner>::lifter(input.as_tensor());
    assert_eq!(lifted.dims(), [1, 2]);
    assert!(lifted.device().same_device(input.device()));
    assert_eq!(data(&lifted), vec![1.0, 2.0]);
    assert!(
        lifted
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .get(&input)
            .is_none()
    );
}
