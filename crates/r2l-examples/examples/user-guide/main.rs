use r2l_api::{LearningSchedule, PPOAlgorithmBuilder};

fn main() {
    let learning_schedule = LearningSchedule::total_step_bound(100000);
    let algorithm_builder =
        PPOAlgorithmBuilder::gym("Pendulum-v1", 10).with_learning_schedule(learning_schedule);
    let mut algorithm = algorithm_builder.build().unwrap();
    algorithm.train().unwrap();
}
