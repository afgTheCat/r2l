# r2l

High-level training builders and default hooks for `r2l`.

The main entry points are `PPOBuilder` and `A2CBuilder`.
They combine environment construction, rollout collection, a Candle or Burn
learning module, scheduling, reporting, and optional best-actor evaluation.

Gymnasium support is opt-in and requires Python 3.11 or newer with the
`gymnasium` package installed:

```toml
r2l = { version = "0.0.2", features = ["gym"] }
```

```rust,no_run
use r2l::{PPOBuilder, TrainingLimit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let builder = PPOBuilder::gym("Pendulum-v1", 4)?
        .with_burn()
        .with_rollout_steps(1024)
        .with_training_limit(TrainingLimit::steps(100_000));
    let mut algorithm = builder.build()?;
    algorithm.train()?;
    Ok(())
}
```

See the [book](https://afgthecat.github.io/r2l/) for a user guide and
[docs.rs](https://docs.rs/r2l/0.0.2/r2l/) for the complete API.
