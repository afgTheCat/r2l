# r2l-core

Core traits and data types shared by the `r2l` reinforcement-learning
workspace.

This crate defines environments and spaces, backend-neutral tensors, actors,
policies, learning modules, trajectory buffers, and the generic on-policy
training loop. It intentionally does not provide a complete end-user training
configuration; use [`r2l-api`](https://crates.io/crates/r2l-api) for high-level
PPO and A2C builders.

API documentation is available on
[docs.rs](https://docs.rs/r2l-core/0.0.2/r2l_core/).
