use r2l_core::tensor::R2lTensor;

use super::*;
use crate::Categorical;

type TestLearner = PolicyValueLearner<Categorical<Mlp>>;

fn learner(split: bool) -> TestLearner {
    let policy = Categorical::new(Mlp::build(&[1, 4, 2], ActivationFunction::Tanh)).unwrap();
    let optimizer = AdamWConfig::new().with_weight_decay(0.0);
    if split {
        PolicyValueLearner::split(
            policy,
            &[1, 4, 1],
            ActivationFunction::Tanh,
            &optimizer,
            0.01,
            &optimizer,
            0.02,
        )
    } else {
        PolicyValueLearner::joint(
            policy,
            &[1, 4, 1],
            ActivationFunction::Tanh,
            &optimizer,
            0.01,
        )
    }
}

fn update(learner: &mut TestLearner, target: f32) {
    let observations = learner.tensor_from_slice(&[0.5, -1.0]).unwrap();
    let actions = Tensor::zeros([2, 1], &observations.device());
    let policy_loss = learner
        .policy()
        .log_probs(observations.clone(), actions)
        .unwrap()
        .mean()
        .reshape([1, 1])
        .neg();
    let value_loss = learner
        .values(observations)
        .unwrap()
        .sub_scalar(target)
        .powf_scalar(2.0)
        .mean()
        .reshape([1, 1]);
    learner
        .update(PolicyValueLosses::new(policy_loss, value_loss))
        .unwrap();
}

fn outputs(learner: &TestLearner) -> Vec<f32> {
    let observations = learner.tensor_from_slice(&[0.5, -1.0]).unwrap();
    let actions = Tensor::zeros([2, 1], &observations.device());
    let log_probs = learner
        .policy()
        .log_probs(observations.clone(), actions)
        .unwrap();
    Tensor::cat(vec![log_probs, learner.values(observations).unwrap()], 1)
        .to_vec()
        .unwrap()
}

#[test]
fn snapshot_records_preserve_the_next_optimizer_update() {
    for split in [false, true] {
        let mut original = learner(split);
        update(&mut original, 2.0);
        update(&mut original, -1.0);
        original.set_learning_rates(0.003, 0.007);

        let path = std::env::temp_dir().join(format!(
            "r2l-burn-snapshot-{}-{split}.bpk",
            std::process::id()
        ));
        match &original {
            PolicyValueLearner::Joint(joint) => joint.to_snapshot().to_file(path.clone()),
            PolicyValueLearner::Split(split) => SplitPolicyValueLeranerSnapshot {
                policy: split.policy.clone().into_record(),
                value_net: split.value_net.clone().into_record(),
                policy_optimizer: split.policy_optimizer.to_record(),
                policy_lr: split.policy_lr,
                value_optimizer: split.value_optimizer.to_record(),
                value_lr: split.value_lr,
            }
            .to_file(path.clone()),
        }

        let reader = burn_pack::Reader::from_file(&path).unwrap();
        let bytes = |name| Bytes::from_bytes_vec(reader.tensor_data(name).unwrap());
        let rate = |name| f64::try_from(*reader.scalars().get(name).unwrap()).unwrap();
        let snapshot = if split {
            PolicyValueLearnerSnapshot::Split(SplitPolicyValueLeranerSnapshot {
                policy: ModuleRecord::from_bytes(bytes("policy")).unwrap(),
                value_net: ModuleRecord::from_bytes(bytes("value_net")).unwrap(),
                policy_optimizer: OptimizerRecord::from_bytes(bytes("policy_optimizer")).unwrap(),
                policy_lr: rate("policy_lr"),
                value_optimizer: OptimizerRecord::from_bytes(bytes("value_optimizer")).unwrap(),
                value_lr: rate("value_lr"),
            })
        } else {
            PolicyValueLearnerSnapshot::Joint(JointPolicyValueSnapshot {
                model: ModuleRecord::from_bytes(bytes("model")).unwrap(),
                optimizer: OptimizerRecord::from_bytes(bytes("optimizer")).unwrap(),
                lr: rate("lr"),
            })
        };
        let mut restored = learner(split).load_snapshot(snapshot);
        std::fs::remove_file(path).unwrap();
        assert_eq!(restored.policy_learning_rate(), 0.003);
        if let PolicyValueLearner::Split(split) = &restored {
            assert_eq!(split.value_lr, 0.007);
        }
        assert_eq!(outputs(&original), outputs(&restored));

        update(&mut original, 0.25);
        update(&mut restored, 0.25);
        for (expected, actual) in outputs(&original).into_iter().zip(outputs(&restored)) {
            assert!((expected - actual).abs() < 1e-6, "{expected} != {actual}");
        }
    }
}
