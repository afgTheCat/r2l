mod dummy_env_holder;

use crate::buffers::step_bound_buffer::StateBuffer;
use r2l_core2::env::{Action, Env, Observation, SnapShot};

trait SequntialStepBoundHooks<Obs: Observation, Act: Action> {
    fn process_last_step<E: Env<Obs = Obs, Act = Act>>(&self, buffers: &mut Vec<StateBuffer<E>>);
    fn process_snapshots(&self, snapshots: Vec<SnapShot<Obs, Act>>) -> Vec<SnapShot<Obs, Act>>;
    // TODO: we might not need these
    fn post_process_hook(&self) {}
}
