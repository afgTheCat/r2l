# r2l-burn

Burn-backed policies and on-policy learning modules for `r2l`.

This crate implements categorical, multi-categorical, Bernoulli, recurrent
categorical, and diagonal Gaussian policies together with joint and split
policy/value learning modules. Most applications select this backend through
`r2l-api`'s `with_burn()` builder method.

API documentation is available on
[docs.rs](https://docs.rs/r2l-burn/0.0.2/r2l_burn/).
