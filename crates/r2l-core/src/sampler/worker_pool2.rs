use crate::{
    distributions::Policy,
    env::Env,
    sampler::{RolloutMode, buffer::ExpandableTrajectoryContainer, worker::WorkerPool},
};

// TODO: do not know if we need this
struct WorkerPool2<E: Env, B: ExpandableTrajectoryContainer<Tensor = E::Tensor>> {
    rollout_mode: RolloutMode,
    worker_pool: WorkerPool<E, B>,
}

impl<E: Env, B: ExpandableTrajectoryContainer<Tensor = E::Tensor>> WorkerPool2<E, B> {
    fn collect<P: Policy<Tensor = E::Tensor> + Clone>(&mut self, policy: P) {
        self.worker_pool.set_policy(policy.clone());
        self.worker_pool.collect(self.rollout_mode);
    }
}
