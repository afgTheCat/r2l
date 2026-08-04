# Examples

The workspace includes runnable examples in `crates/r2l-examples`. They are
intended to show complete setups that can be inspected, copied, and changed
without reading every lower-level crate first.

Run examples from the workspace root:

```sh
cargo run -p r2l-examples --example <name>
```

The examples in this section focus on two common starting points:

- [Environment building](./examples/env_building.md): implementing `Env`,
  using `EnvBuilder`, and plugging Gymnasium environments into builders.
- [Algorithms](./examples/algorithms.md): configuring PPO and A2C runs with
  backends, rollout bounds, training schedules, reporting, and artifacts.

Gymnasium-backed examples require Python 3.11 or newer with `gymnasium`
installed in the Python environment used by the process.
