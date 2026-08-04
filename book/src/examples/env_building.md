# Environment building

Algorithms are built from an `EnvBuilder`, not from a single environment
instance. This lets samplers create one environment per worker and keeps
environment state local to the worker that steps it.

The `env_building` example shows the supported construction styles:

- implementing `Env` for a custom environment;
- implementing `EnvBuilder` for a custom builder type;
- passing a closure or function that returns an environment;
- using `GymEnvBuilder` directly;
- using `PPOAlgorithmBuilder::gym` for Gymnasium environment ids.

Run it from the workspace root:

```sh
cargo run -p r2l-examples --example env_building
```

The full example is:

```rust,noplayground
{{#include ../../../crates/r2l-examples/examples/env_building/main.rs:env_builders}}
```

For real environments, make sure `env_description` accurately describes the
flattened observation and action spaces. The policy builder uses those spaces to
choose the policy distribution and network dimensions.
