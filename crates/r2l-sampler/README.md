# r2l-sampler

Inline and threaded rollout samplers for `r2l`.

`DirectSampler` lets workers write transitions directly to their output
buffers. `StagedSampler` receives transitions from workers and can optionally
maintain shared running observation statistics before committing trajectories.
Hook implementations control the step or episode bounds used for each rollout.

API documentation is available on
[docs.rs](https://docs.rs/r2l-sampler/0.0.3/r2l_sampler/).
