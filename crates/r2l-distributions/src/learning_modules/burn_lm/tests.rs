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
        original.to_snapshot().to_file(path.clone());

        let reader = burn_pack::Reader::from_file(&path).unwrap();
        let bytes = |name| Bytes::from_bytes_vec(reader.tensor_data(name).unwrap());
        let rate = |name| f64::try_from(*reader.scalars().get(name).unwrap()).unwrap();
        let optimizer = if split {
            OptimizerKind::Split {
                policy: OptimizerRecord::from_bytes(bytes("policy_optimizer")).unwrap(),
                policy_lr: rate("policy_lr"),
                value: OptimizerRecord::from_bytes(bytes("value_optimizer")).unwrap(),
                value_lr: rate("value_lr"),
            }
        } else {
            OptimizerKind::Joint {
                optimizer: OptimizerRecord::from_bytes(bytes("optimizer")).unwrap(),
                lr: rate("lr"),
            }
        };
        let snapshot = PolicyValueLearnerSnapshot {
            model: ModuleRecord::from_bytes(bytes("model")).unwrap(),
            optimizer,
        };
        let mut restored = learner(split).load_snapshot(snapshot);
        std::fs::remove_file(path).unwrap();
        assert_eq!(restored.policy_learning_rate(), 0.003);
        if let OptimizerKind::Split { value_lr, .. } = &restored.optimizer.inner {
            assert_eq!(*value_lr, 0.007);
        }
        assert_eq!(outputs(&original), outputs(&restored));

        update(&mut original, 0.25);
        update(&mut restored, 0.25);
        for (expected, actual) in outputs(&original).into_iter().zip(outputs(&restored)) {
            assert!((expected - actual).abs() < 1e-6, "{expected} != {actual}");
        }
    }
}
