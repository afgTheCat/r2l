# r2l

High-level training builders and default hooks for `r2l`.

The main entry points are `PPOAlgorithmBuilder` and `A2CAlgorithmBuilder`.
They combine environment construction, rollout collection, a Candle or Burn
learning module, scheduling, reporting, and optional best-actor evaluation.

```rust,no_run
use r2l::{LearningSchedule, PPOAlgorithmBuilder};

let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
    .with_burn()
    .with_rollout_steps(1024)
    .with_learning_schedule(LearningSchedule::total_step_bound(100_000));
let mut algorithm = builder.build().unwrap();
algorithm.train().unwrap();
```

See the [book](https://afgthecat.github.io/r2l/) for a user guide and
[docs.rs](https://docs.rs/r2l/0.0.2/r2l/) for the complete API.
