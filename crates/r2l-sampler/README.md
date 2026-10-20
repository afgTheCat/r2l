# r2l-sampler

Inline and threaded rollout samplers for `r2l`.

`RolloutSampler` supports two collection strategies: direct workers write
transitions into their output buffers, while staged workers share observation
normalization coordinated between steps. Both use the same `SamplerHook`
interface to control step or episode bounds and transform collected buffers.

Use `RolloutSampler::build_direct` for raw observations or
`RolloutSampler::build_staged` with a normalizer. Evaluation uses a read-only
handle to the training normalizer's statistics.

API documentation is available on
[docs.rs](https://docs.rs/r2l-sampler/0.0.3/r2l_sampler/).
