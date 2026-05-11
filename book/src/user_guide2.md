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
    let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4);
    let mut algo = builder.build().unwrap();
    algo.train().unwrap();
}
```

On top of `PPOAlgorithmBuilder`, **r2l** also have `PPOAlgorithmBuilder`. The
first argument for the builder is the name of the environment, while the second
one is the number of environments to be run in paralell.

## Choosing the backend

Algorithm builder can be choosen with the `with_burn` or `with_canlde` method.
The current `burn` backend defaults to `NdArray`.

```rust
let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4).with_burn().with_candle(candle_core::Device::Cpu);
```

## Choosing the execution strategy

Sampling trajectories can benefit from using worker threads instead using only a
single thread. To do that, you might specify the execution mode by setting the
sampler execution mode:

```rust
let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
  .with_execution_mode(SamplerExecutionMode::Thread)
  .with_execution_mode(SamplerExecutionMode::Vec);
```

Note that for `gymnasium` environments, it makes no sense to run
`SamplerExecutionMode::Thread`, as the GIL won't allow paralell execution.

## Learning scheduling

By learning schedule we mean the threshold that needs to be reached in order for
learaning to stop. **r2l** currently supports two kinds, one that is based on
rollouts, and one that is based on the total number of steps elapsed. You can
set them by

```rust
let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
    .with_learning_schedule(LearningSchedule::rollout_bound(10))
    .with_learning_schedule(LearningSchedule::total_step_bound(1000));
```

Note that `LearningSchedule::TotalStepBound` will count the total steps across
all environments. That is, if you have `n` envs and `10*n` as the step bound,
each environment will step 10 times in total.

## Rollout bounds

Similiarly to learning schedule, you can also change how rollouts should be
collected. **r2l** currently supports one that is based on the number of steps
taken, and one that is based on the number of episodes that elapsed.

```rust
let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
    .with_rollout_bound(EpisodeTrajectoryBound::new(10))
    .with_rollout_bound(StepTrajectoryBound::new(1000));
```

Similarly to learning scheudle, `StepTrajectoryBound` counts the total step for
all environments. Setting the rollout bound also controls the shape of the
`TrajectoryContainer`. Generally speaking, `EpisodeTrajectoryBound` should be
faster, as we can pre allocate the right amount of storage exactly once, while
`EpisodeTrajectoryBound` forces us to use a `Vec` as storage, which might need
to allocate multiple times.

## Logging training results

Results can be logged two ways currently: either to the standard output or to a
channel. Logging to the standard output is enabled by default, while channel
logging can be set by providing the channel.

```rust
let builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 4)
    .with_log_progress(false)
    .with_reporter(Some(tx));
```

For most usecases, the deafult setup should be fine, but for certain usecases
you might want the reporter to be present and not want the stdout to be kept
clean (such as in the case of a tui application).

## Saving the best performing agent
