# r2l-sampler

Inline and threaded rollout samplers for `r2l`.

`R2lSampler` stores raw environment observations and rewards.
`R2lNormalizedSampler` optionally maintains shared running observation
statistics and exposes normalized trajectories. Hook implementations control
the step or episode bounds used for each rollout.

API documentation is available on
[docs.rs](https://docs.rs/r2l-sampler/0.0.2/r2l_sampler/).
