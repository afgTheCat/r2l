# r2l-agents

Lower-level on-policy algorithm implementations for `r2l`.

The crate contains PPO, A2C, and VPG learning logic plus their hook interfaces
and shared rollout-processing utilities. Most applications should use the
builders and default hooks from
[`r2l`](https://crates.io/crates/r2l).

API documentation is available on
[docs.rs](https://docs.rs/r2l-agents/0.0.3/r2l_agents/).
