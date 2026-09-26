# r2l-distributions

Shared categorical, diagonal Gaussian, multi-Bernoulli, multi-categorical and
composite policies, with Burn and Candle networks and on-policy learners.
This crate replaces the former backend integration crates. It depends directly
on Burn, Candle and `r2l-core`.

Use `r2l::PPOBuilder` or `r2l::A2CBuilder` for complete training loops. For custom
learners, see the examples in `learning_modules::burn_lm` and
`learning_modules::candle_lm`.

## Tensor contracts

`Policy`, `ValueFunction` and `OnPolicyLearner` live in `r2l-core` and are reexported
here. Observations and actions use `[batch, features]`; log probabilities, values,
returns and advantages use `[batch, 1]`. A single actor call accepts `[1, features]`
and returns `[1, actions]`. The `r2l` runtime adapts environment tensors at its
sampling and learning boundaries.

Burn Gaussian policies register `log_std` through `Param<Tensor<B, 2>>`. Candle
policies obtain it from the optimizer's `VarMap` before constructing the optimizer.
For reproducible Candle initialization, use `networks::seeded_var_builder` with
`r2l_core::rng::set_seed`.

Candle inference policies detach their outputs and share live parameter storage.
Complete rollout collection before optimizer updates, as the built-in loop does.

## Networks, artifacts and features

Burn supports MLP and CNN configurations; Candle supports MLP configurations.
Built-in policies implement `ToSafetensors`.

The `simd` feature enables Burn SIMD support. The `cuda` feature enables Candle
CUDA support. These features are forwarded by the corresponding `r2l` features.
