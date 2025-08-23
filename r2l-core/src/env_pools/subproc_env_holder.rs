use crate::env_pools::RolloutMode;
use crate::env_pools::{EnvHolder, SequentialVecEnvHooks};
use crate::ipc::{PacketToReceive, PacketToSend, receive_packet, send_packet};
use crate::{distributions::Distribution, utils::rollout_buffer::RolloutBuffer};
use bincode::{Decode, Encode};
use candle_core::{Error, Result, Tensor};
use interprocess::local_socket::{
    GenericNamespaced, ListenerOptions, Stream, ToNsName, traits::ListenerExt,
};
use std::{io::BufReader, process::Child};

struct SubprocessEnvHandle {
    child: Child,
    conn: BufReader<Stream>,
}

pub struct SubprocHolder {
    handles: Vec<SubprocessEnvHandle>,
}

impl SubprocHolder {
    fn shutdown_subprocesses<
        D: Distribution<Action = Tensor, Observation = Tensor, Entropy = Tensor> + Decode<()> + Encode,
    >(
        &mut self,
    ) {
        for SubprocessEnvHandle { conn, .. } in self.handles.iter_mut() {
            let packet_sent: PacketToSend<D> = PacketToSend::Halt;
            send_packet(conn, packet_sent);
            let packet_received: PacketToReceive<D> = receive_packet(conn);
            assert!(matches!(packet_received, PacketToReceive::Halting));
        }
    }

    pub fn build(socket_name: &str, num_proc: usize) -> Result<Self> {
        let socket_ns_name = socket_name
            .to_ns_name::<GenericNamespaced>()
            .map_err(Error::wrap)?;
        let opts = ListenerOptions::new().name(socket_ns_name);
        let listener = opts.create_sync().map_err(Error::wrap)?;
        let mut handles = vec![];
        for _ in 0..num_proc {
            // TODO: we should make the args type safe
            let child = std::process::Command::new("cargo")
                .args([
                    "run",
                    "--bin",
                    "-p",
                    "subproc_env",
                    "--bin",
                    "subproc_env",
                    "--",
                    "--env-construction-method",
                    "gym-env",
                    "--env-name",
                    "Pendulum-v1",
                    "--steps",
                    "1024",
                    "--socket-name",
                    socket_name,
                ])
                .spawn()
                .map_err(Error::wrap)?;
            let conn = listener.incoming().next().unwrap().map_err(Error::wrap)?;
            let conn = BufReader::new(conn);
            handles.push(SubprocessEnvHandle { child, conn });
        }
        Ok(Self { handles })
    }
}

impl EnvHolder for SubprocHolder {
    fn num_envs(&self) -> usize {
        todo!()
    }

    fn sequential_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
        hooks: &mut dyn SequentialVecEnvHooks,
    ) -> Result<Vec<RolloutBuffer>> {
        todo!()
    }

    fn async_rollout<D: Distribution>(
        &mut self,
        distr: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>> {
        // for SubprocessEnvHandle { conn, .. } in self.handles.iter_mut() {
        //     let packet = PacketToSend::StartRollout {
        //         distribution,
        //         rollout_mode,
        //     };
        //     send_packet(conn, packet);
        // }
        // let mut rollouts = vec![];
        // // collecting all the environments in a loop is fine, since the bulk of the computation
        // // will happen in a different process anyways
        // for SubprocessEnvHandle { conn, .. } in self.handles.iter_mut() {
        //     let recieved_packet: PacketToReceive<DistributionKind> = receive_packet(conn);
        //     let PacketToReceive::RolloutResult { rollout } = recieved_packet else {
        //         unreachable!()
        //     };
        //     rollouts.push(rollout);
        // }
        // Ok(rollouts)

        todo!()
    }
}
