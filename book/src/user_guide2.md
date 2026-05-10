## Getting started

At it's core, **r2l** aims to be a reinforcement learning library implementing
some of the more popular algorithms. Architecturally, **r2l** is modular,
minimal and hookable. Most users would want to start with the **r2l-api**, which
exposes the most important lower level structures, as well as concrete hook
implementations. The goal is to hide complexity behind an easy to configure
builder style API. As an example, if you have `gymnasium` installed on your
machine, you can get started like so:

```rust
use r2l_api::{LearningSchedule, PPOAlgorithmBuilder, StepTrajectoryBound};

fn main() {
    let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
        .with_clip_range(0.2)
        .with_lambda(0.95)
        .with_gamma(0.9)
        .with_learning_rate(0.001)
        .with_total_epochs(10)
        .with_bound(StepTrajectoryBound::new(1024))
        .with_learning_schedule(LearningSchedule::total_step_bound(100000));
    let mut algo = builder.build().unwrap();
    algo.train().unwrap();
}
```
