use super::EnvPool;
use crate::{distributions::Distribution, env::RolloutMode, utils::rollout_buffer::RolloutBuffer};
use bincode::{Decode, Encode};
use candle_core::{Error, Result};
use interprocess::local_socket::{
    GenericNamespaced, ListenerOptions, Stream, ToNsName, traits::ListenerExt,
};
use std::{
    io::{BufReader, Read, Write},
    process::Child,
};

#[derive(Debug, Encode)]
pub enum PacketToSend<'a, D: Distribution> {
    // Send command to halt training
    Halt,
    // Ack halting command
    Halting,
    // Train an epoch with current logits
    StartRollout {
        distribution: &'a D,
        rollout_mode: RolloutMode,
    },
    // Return the trained amount
    RolloutResult {
        rollout: RolloutBuffer,
    },
}

#[derive(Debug, Decode)]
pub enum PacketToReceive<D: Distribution> {
    // Send command to halt training
    Halt,
    // Ack halting command
    Halting,
    // Train an epoch with current logits
    StartRollout {
        distribution: D,
        rollout_mode: RolloutMode,
    },
    // Return the trained amount
    RolloutResult {
        rollout: RolloutBuffer,
    },
}

// Custom low level protocol to send data
pub fn send_packet<D: Encode>(conn: &mut BufReader<Stream>, packet: D) {
    let payload = bincode::encode_to_vec(packet, bincode::config::standard()).unwrap();
    let payload_len = (payload.len() as u32).to_be_bytes();
    conn.get_mut().write_all(&payload_len).unwrap();
    conn.get_mut().write_all(&payload).unwrap();
    conn.get_mut().flush().unwrap();
}

// Custom low level protocol to receive data
pub fn receive_packet<D: Decode<()>>(conn: &mut BufReader<Stream>) -> D {
    let mut content_len = [0u8; 4];
    conn.read_exact(&mut content_len).unwrap();
    let len = u32::from_be_bytes(content_len);
    let mut buffer = vec![0u8; len as usize];
    conn.read_exact(&mut buffer).unwrap();
    let (packet, _): (D, _) =
        bincode::decode_from_slice(&buffer, bincode::config::standard()).unwrap();
    packet
}

struct SubprocessEnvHandle {
    child: Child,
    conn: BufReader<Stream>,
}

pub struct SubprocessingEnv {
    handles: Vec<SubprocessEnvHandle>,
}

// TODO: This needs to be reconsidered
impl EnvPool for SubprocessingEnv {
    fn collect_rollouts<D: Distribution>(
        &mut self,
        distribution: &D,
        rollout_mode: RolloutMode,
    ) -> Result<Vec<RolloutBuffer>>
// where
//     D: Decode<()> + Encode,
    {
        todo!()
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
    }

    fn env_description(&self) -> super::EnvironmentDescription {
        todo!()
    }
}

impl SubprocessingEnv {
    // Graceful shutdown
    fn shutdown_subprocesses<D: Distribution + Decode<()> + Encode>(&mut self) {
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
