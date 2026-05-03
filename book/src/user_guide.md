## Too long, don't care

If you want a complete example first, jump to [Examples](#examples). The rest of
this guide explains the concepts behind those examples and the main builder APIs
exposed by `r2l-api`.

## Overview

This guide covers the main pieces you need to get started with **r2l**:

- how to define your own environment or use an existing one provided by
  `gymnasium`
- how to construct and configure training through the builder API
- the core terms used by the on-policy training stack

For most users, `r2l-api` should be the only `r2l` crate they need to interact
with directly. If you do not already have an environment, `r2l-gym` provides
integration with `gymnasium`. At the moment, `r2l-api` focuses on on-policy
algorithms.

The `r2l-api` crate is intentionally high level. It builds on lower-level `r2l`
crates and aims to provide an ergonomic training surface without exposing every
internal abstraction up front. For more details on the underlying architecture,
see the [On policy algorithms](./on_policy_algorithms.md) chapter.

## Environments

Every training run starts with an environment. In `r2l`, environments implement
the `Env` trait.

```rust,noplayground
{{#include ../../crates/r2l-core/src/env.rs:env}}
```

The `Env` trait is neither `Sync` nor `Send`. Some environments cannot offer
those guarantees. A typical example is a simulator backed by a dynamically
loaded shared library with global state.

Because `r2l` supports multithreaded rollout collection, environments are
usually passed around as builders instead of concrete instances. The
`EnvBuilder` trait lets higher-level builders create environments on the thread
where they will actually run.

```rust,noplayground
{{#include ../../crates/r2l-core/src/env.rs:env_builder}}
```

In practice, the simplest `EnvBuilder` is often just a closure or function that
creates a fresh environment instance.

```rust
let number_of_environments = 10;
let env_builder = || Ok(MyEnv);
let ppo_builder = PPOAlgorithmBuilder::new(env_builder, number_of_environments);
```

That is enough to start using the algorithm builders shown in the next section.
For more complete examples of `Env` and `EnvBuilder`, see
[Examples: Environments](#examples-environments).

### Gym environments

If you are using `gymnasium`, `r2l-gym` provides `GymEnv` and `GymEnvBuilder`.
The algorithm builders in `r2l-api` expose a dedicated `gym` constructor, which
is usually the shortest path to a working setup.

```rust
// anything that implements the Into<GymEnvBuilder> can be used with `gym`
let ppo_builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 10);
```

Currently, `Discrete` and `Box` action spaces are supported. Note that Python's
GIL limits true parallelism when sampling Gym environments from multiple
threads.

## Algorithm builders

Algorithm builders are the main entry point for most users. They combine
environment setup, rollout collection, and policy updates into one builder.

At the moment, `r2l-api` exposes the following on-policy algorithm builders:

- `PPOAlgorithmBuilder`
- `A2CAlgorithmBuilder`

If you are getting started, pick one of these first. Only drop down to sampler
or agent builders when you need lower-level control.

The standard workflow is:

1. define an environment or environment builder
2. choose an algorithm builder
3. configure the builder
4. build and train

If you already have an `EnvBuilder`, you can start directly with an algorithm
builder:

```rust
let number_of_environments = 10;
let env_builder = || Ok(MyEnv);
let ppo_builder = PPOAlgorithmBuilder::new(env_builder, number_of_environments);
```

Or if you are using `gymnasium`:

```rust
// anything that implements the Into<GymEnvBuilder> can be used with `gym`
let ppo_builder = PPOAlgorithmBuilder::gym("Pendulum-v1", 10);
```

Most users can stop at this level. The next sections describe the lower-level
building blocks that algorithm builders compose internally.

## Core concepts of on-policy training in **r2l**

Inside `r2l`, an on-policy training loop has two main stages:

- collecting samples using the `Sampler`
- updating the policy and value function using the `Agent`

The `Algorithm` coordinates those stages and runs until a stopping condition has
been reached, such as a fixed number of rollouts, steps, or episodes.

![A simplfied overview of On Policy algorithms](./images/simplified_onpolicy_learning_loop.png)

`r2l-api` exposes builders for samplers, agents, and algorithms. Algorithm
builders are the highest-level interface. Sampler and agent builders exist for
users who want to customize rollout collection and learning separately.

| Builder               | Builder type      | Produces                           | Notes                                                                                                       |
| --------------------- | ----------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `SamplerBuilder`      | Sampler builder   | `R2lSampler`                       | Useful when you want to pair rollout collection with your own agent or algorithm setup.                     |
| `PPOAgentBuilder`     | Agent builder     | `PPOBurnAgent` or `PPOCandleAgent` | Builds the policy update component only; pair it with a sampler when not using `PPOAlgorithmBuilder`.       |
| `A2CAgentBuilder`     | Agent builder     | `A2CBurnAgent` or `A2CCandleAgent` | Similar to `PPOAgentBuilder`, but for A2C training.                                                         |
| `PPOAlgorithmBuilder` | Algorithm builder | A configured PPO `Algorithm`       | The highest-level PPO entry point; combines environment setup, sampler construction and agent construction. |
| `A2CAlgorithmBuilder` | Algorithm builder | A configured A2C `Algorithm`       | The highest-level A2C entry point; usually the simplest way to start training with A2C.                     |

## Sampler

This is a lower-level customization API. You do not need it to get started with
`PPOAlgorithmBuilder` or `A2CAlgorithmBuilder`.

The default on-policy sampler in `r2l` is `R2lSampler`, which can be constructed
with
[`SamplerBuilder`](https://docs.rs/r2l-api/0.0.2-rc1/r2l_api/struct.SamplerBuilder.html).
To build a sampler, provide an environment builder and the number of
environments to spawn. For each worker, an `Actor` derived from the current
policy steps the environment until a trajectory bound is reached.

You can also choose the execution mode. By default, environments are stepped
sequentially on the current thread. To run them on worker threads, switch the
execution mode to `Thread`.

```rust
let gym_env_builder = GymEnvBuilder::new("Pendulum-v1");
let sampler_builder = SamplerBuilder::<GymEnvBuilder>::new(gym_env_builder, 10)
    .with_execution_mode(SamplerExecutionMode::Vec)
    .with_execution_mode(SamplerExecutionMode::Thread)
    .with_bound(EpisodeTrajectoryBound::new(10))
    .with_bound(StepTrajectoryBound::new(1000));
let sampler = sampler_builder.build();
```

## Agent builders

This is also a lower-level customization API. Use these builders when you want
to configure the learning step separately from the sampler or algorithm.

`A2CAgentBuilder` and `PPOAgentBuilder` produce different agents, but they share
many configuration options. Those common parameters are documented
[here](https://docs.rs/r2l-api/0.0.2-rc1/r2l_api/struct.OnPolicyAgentBuilder.html).

### A2C agent builder

A2C is a synchronous, deterministic variant of A3C. `A2CAgentBuilder` exposes no
extra parameters beyond the shared on-policy agent configuration. For more
background, see the [paper](https://arxiv.org/abs/1602.01783).

```rust
// The standard Algorithm builder options
let a2c_algo = A2CAgentBuilder::new(10)
    .with_burn()
    .with_normalize_advantage(true)
    .with_entropy_coeff(0.)
    .with_vf_coeff(None)
    .with_policy_hidden_layers(vec![32, 32])
    .build(10, 2, ActionSpaceType::Discrete);
```

### PPO agent builder

PPO combines ideas from A2C and TRPO and adds PPO-specific configuration such as
clipping and target KL settings. For more background, see the
[paper](https://arxiv.org/abs/1707.06347).

```rust
// target_kl and clip range are ppo specific
let ppo_algo = PPOAgentBuilder::new(10)
    .with_target_kl(Some(0.3))
    .with_clip_range(0.5);
```

## Examples {#examples}

The examples below show the most common `r2l-api` workflows. If you want to go
beyond the high-level API, start with [architecture](./architecture.md) and the
[On policy algorithms](./on_policy_algorithms.md) chapter.

### Environments {#examples-environments}

The below example shows implementing the `Env` trait and different
implementations of the `EnvBuilder`.

```rust
{{#include ../../crates/r2l-examples/examples/env_building/main.rs:env_builders}}
```

### A2C {#examples-a2c}

This example shows how to train with the A2C algorithm builder.

```rust
{{#include ../../crates/r2l-examples/examples/a2c/main.rs:a2c}}
```

### PPO {#examples-ppo}

This example shows how to train with the PPO algorithm builder.

```rust
{{#include ../../crates/r2l-examples/examples/ppo/main.rs:ppo}}
```
