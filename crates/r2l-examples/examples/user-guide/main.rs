use r2l_api::{LearningSchedule, PPOAlgorithmBuilder};

fn main() {
    // define some hyper parameters
    let learning_schedule = LearningSchedule::total_step_bound(100000);
    // set those hyper parmeters
    let algorithm_builder =
        PPOAlgorithmBuilder::gym("Pendulum-v1", 10).with_learning_schedule(learning_schedule);
    // build the algorithm
    let mut algorithm = algorithm_builder.build().unwrap();
    // train the algorithm
    algorithm.train().unwrap();

    // SAVE THE AGENT ??
    // LOAD THE AGENT ??
    // CONTINUE TRAINING ?? (BY CREATING AN ALGORITHM)
    // RUN INFERENCE ??
}
