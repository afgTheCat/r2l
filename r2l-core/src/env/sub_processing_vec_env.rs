use super::{EnvPool, RolloutMode};
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use candle_core::Result;
use interprocess::local_socket::Stream;
use std::{io::BufReader, process::Child};

#[allow(dead_code)]
struct SubprocessEnvHandle {
    child: Child,
    conn: BufReader<Stream>,
}

#[allow(dead_code)]
struct SubPorcessingEnv {
    envs: Vec<SubprocessEnvHandle>,
    rollout_mode: RolloutMode,
}

impl EnvPool for SubPorcessingEnv {
    fn collect_rollouts<D: Distribution>(&self, _distribution: &D) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }
}
