> [!WARNING]  
> **Pre-Alpha:** This library is under active development. Current APIs are almost surely going to change in the future.

The current user guide only covers on policy algorithms. This page focuses on key concepts in  `r2l` as well as covering 
the user facing APIs/capabilities.

## Too long, don't care

Check out the [ppo](#examples-ppo) or [a2c](#examples-a2c) examples.

## Crates
R2l composes multiple crates, of which the following are of interest for the end user.

- `crates/r2l-core`: core RL abstractions and on-policy algorithms
- `crates/r2l-gym`: wrappers around Gym-style environments
- `crates/r2l-api`: shared API-facing types and interfaces

If you believe your needs are common yet not covered by the `r2l-api`, feel free to open an issue/pr.

## Components

### Environments
In order to use `r2l` an environments needs to be constructed. The environment trait is defined as.

```rust
{{#include ../../crates/r2l-core/src/env.rs:env}}
```

Conceptually, there could be environments that are neither `Sync` nor `Send`. An example could be dynamically loading a shared 
library that has no constructors and uses global variables, like most embedded firmwares. Working with such environments
forces us to construct the environment on the worker thread. The solution for this is using environment builders.

```rust
{{#include ../../crates/r2l-core/src/env.rs:env_builder}}
```

The simplest `EnvBuilder`  possible is a closure or function that returns a new instance of the environment. As an example,
if `MyEnv` implements the `Env` trait, the below closure can work as an `EnvBuilder`.

```rust
let env_builder = || Ok(MyEnv);
let ppo_builder = PPOAlgorithmBuilder::new(env_builder, 10);
```

For more examples on how to implement the `Env` and `EnvBuilder` trait, check the [environments](#examples-environments) section 
of the examples.

#### Gym environments
Gym environments from gymnasium are implemented in `gym-env`. Currently `Discrete` and `Box` action spaces are implemented. A 
gym environment is wrapped by the `GymEnv` struct, while `GymEnvBuilder` can be used to construct a `gym` environment. Algorithms 
exposed by `r2l-api` handle gym environments through a dedicated `gym` constructor.

```rust
let ppo_builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 10);
```

A sidenote on `gym` environments: while it is possible to use a `ThreadEnvironment`, thanks to the GIL, true paralellism is not
going to happen.

### Sampler
A sampler is responsible for collecting rollout trajectories. `r2l` comes pre-packaged with it's own sampler implementation.
You would probably not need to implement your own sampler, but knowing it's capabilities is useful.

| Feature                   | Status  | Description                                                          | Blocked on    |
| ------------------------- | ------- | -------------------------------------------------------------------- | ------------- |
| Episode based sampling    | ✅      | Collects trajectories based the number of episodes completed         |               |
| Steps based sampling      | ✅      | Collects trajectories based the number of steps taken                |               |
| Observation normalization | ❌      | Normalizes observations using rms                                    | Sampler hooks |
| Reward normalization      | ❌      | Normalized rewards using rms                                         | Sampler hooks |
| Vec environments          | ✅      | Trajectories are collected in sequentsequentially on the same thread |               |
| Thread environments       | ✅      | Trajectories are collected in paralell using multi threading         |               |

Samplers can be constructed accordingly.

```rust
let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
let sampler_builder = SamplerBuilder::<GymEnvBuilder>::new(gym_env_builder, 10)
    .with_location(Location::Vec)
    .with_location(Location::Thread)
    .with_bound(EpisodeTrajectoryBound::new(10))
    .with_bound(StepTrajectoryBound::new(1000));
let sampler = sampler_builder.build();
```

### Agents

An agent is responsible for learning based on the trajectories collected. Currently we support PPO and A2C.

#### PPO
```rust
let agents = PPOAgentBuilder::new(10)
    .with_burn()
    .with_candle(Device::Cpu)
    .with_entropy_coeff(0.1)
    .with_vf_coeff(Some(0.1))
    .with_target_kl(Some(0.5))
    .with_gradient_clipping(Some(1.))
    .with_clip_range(0.5)
    .with_gamma(0.98)
    .with_lambda(0.9)
    .with_sample_size(32)
    .with_policy_hidden_layers(vec![32, 32])
    .with_value_hidden_layers(vec![32, 32])
    .with_learning_rate(3e-4)
    .with_beta1(0.9)
    .with_beta2(0.999)
    .with_epsilon(1e-5)
    .with_weight_decay(1e-4)
    .with_joint(
        None,
        ParamsAdamW {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-5,
            weight_decay: 1e-4,
        },
    );
```

TODO explanation + table

#### A2C
TODO usage + explanation + table

### On policy algorithm

The on policy algorithm connects uses a `sampler` and an `agent` to coordinate learning. Currently we have the `PPOAlgorithmBuilder` 
and the `A2CAlgorithmBuilder` structs. The `sampler` and `agent` builder methods are exposed just like.

## Examples

Below are some examples how to use `r2l-api`, which provides builders and APIs for the most commonly used usecases.
R2l is highly extensible, and if you are curious about the how to extend it/use it, start at [architecture](./architecture.md). 

### Environments {#examples-environments}

The below example shows implementing the `Env` trait and different implementations of the `EnvBuilder`.

```rust
{{#include ../../crates/r2l-examples/examples/env_building/main.rs:env_builders}}
```

### PPO {#examples-ppo}

```rust
{{#include ../../crates/r2l-examples/examples/ppo/main.rs:ppo}}
```

### A2C {#examples-a2c}

```rust
{{#include ../../crates/r2l-examples/examples/a2c/main.rs:a2c}}
```

