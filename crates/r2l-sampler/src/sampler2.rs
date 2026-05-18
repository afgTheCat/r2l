// This is the new sampler impl with hooks. The pattern I seem to be settling on
// in having a core and the hooks around it

use r2l_core::env::Env;

use crate::{R2lSampler, RolloutBound};

trait NewSamplerHook {}

struct NewSampler<E: Env, BD: RolloutBound<Tensor = E::Tensor>> {
    sampler: R2lSampler<E, BD>,
}

impl<E: Env, BD: RolloutBound<Tensor = E::Tensor>> NewSampler<E, BD> {}
