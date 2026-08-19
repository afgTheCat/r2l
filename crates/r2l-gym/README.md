# r2l-gym

Gymnasium environment adapters for `r2l`.

`GymEnv` wraps a Python Gymnasium environment and implements `r2l_core::Env`.
`GymEnvBuilder` constructs named environments for the high-level PPO and A2C
builders. Discrete spaces with `start = 0`, plus Box, MultiDiscrete,
MultiBinary, Tuple, and Dict spaces are supported. Non-zero Discrete `start`
values are not currently supported.

The crate uses PyO3's Python 3.11 stable ABI. The `gymnasium` package must be
installed in the Python environment used at runtime.

API documentation is available on
[docs.rs](https://docs.rs/r2l-gym/0.0.2/r2l_gym/).
