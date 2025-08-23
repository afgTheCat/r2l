use r2l_core2::env::{Action, Observation, SnapShot};

mod dummy_env_holder;

trait SequntialStepBoundHooks<Obs: Observation, Act: Action> {
    fn process_snapshots(&self, snapshots: Vec<SnapShot<Obs, Act>>) -> Vec<SnapShot<Obs, Act>>;
    // TODO: we might not need these
    fn post_process_hook(&self) {}
}
