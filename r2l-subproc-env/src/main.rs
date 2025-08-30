// Eventually we want to load the environments as dynamic libraries via libloading or dlopen-rs
// and supported gym env as well. Also we can experiment with fork/clone whatever on linux

use bincode::{Decode, Encode};
use candle_core::{Device, Error, Result, Tensor};
use clap::Parser;
use interprocess::local_socket::{
    GenericNamespaced, Stream, ToNsName, traits::Stream as StreamTrait,
};
use r2l_core::{
    distributions::{Distribution, DistributionKind},
    env::Env,
    ipc::{PacketToReceive, PacketToSend, receive_packet, send_packet},
    numeric::Buffer,
    sampler::{
        DistributionWrapper,
        trajectory_buffers::variable_size_buffer::VariableSizedTrajectoryBuffer,
    },
};
use r2l_gym::GymEnv;
use std::io::BufReader;

#[derive(Parser, Debug, Clone)]
enum EnvConstructionMethod {
    GymEnv,
    DyLib,
}

impl From<String> for EnvConstructionMethod {
    fn from(value: String) -> Self {
        match value.as_str() {
            "gym-env" => Self::GymEnv,
            "dy-lib" => Self::DyLib,
            _ => unreachable!(),
        }
    }
}

#[derive(Parser, Debug, Clone)]
enum RolloutType {
    EpisodeBound,
    StepBound,
}

impl From<String> for RolloutType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "episode-bound" => Self::EpisodeBound,
            "step-bound" => Self::StepBound,
            _ => unreachable!(),
        }
    }
}

#[derive(Parser, Debug, Clone)]
enum DeviceType {
    CPU,
    CUDA,
}

impl From<String> for DeviceType {
    fn from(value: String) -> Self {
        match value.as_str() {
            "cpu" => Self::CPU,
            "cuda" => Self::CUDA,
            _ => unreachable!(),
        }
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    env_construction_method: EnvConstructionMethod,

    #[arg(long)]
    env_name: String,

    #[arg(long)]
    steps: usize,

    #[arg(long)]
    socket_name: String,
}

pub struct Rollout<E: Env> {
    conn: BufReader<Stream>,
    trajectory_buffer: VariableSizedTrajectoryBuffer<E>,
    device: Device,
}

impl<E: Env<Tensor = Buffer>> Rollout<E> {
    fn handle_packet<D: Distribution<Tensor = Tensor> + Decode<()> + Encode>(
        &mut self,
    ) -> Result<bool> {
        let packet: PacketToReceive<D> = receive_packet(&mut self.conn);
        match packet {
            PacketToReceive::Halt => {
                send_packet(&mut self.conn, PacketToSend::<D>::Halting);
                Ok(false)
            }
            PacketToReceive::StartRollout {
                distribution,
                rollout_mode,
            } => {
                // FIXME: we should act as the sampler like we did the threads. This is not used
                // currently but should be added in the future
                let distribution: DistributionWrapper<D, E> =
                    DistributionWrapper::new(&distribution);
                self.trajectory_buffer
                    .step_with_epiosde_bound(&distribution, 1024);
                let packet: PacketToSend<D> = PacketToSend::RolloutResult {
                    rollout: self.trajectory_buffer.to_rollout_buffer().convert(),
                };
                send_packet(&mut self.conn, packet);
                Ok(true)
            }
            _ => unreachable!(),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match &args.env_construction_method {
        EnvConstructionMethod::GymEnv => {
            let env = GymEnv::new(&args.env_name, None);
            let socket_name = args
                .socket_name
                .to_ns_name::<GenericNamespaced>()
                .map_err(Error::wrap)?;
            let conn = Stream::connect(socket_name).unwrap();
            let conn = BufReader::new(conn);
            let mut rollout = Rollout {
                conn,
                trajectory_buffer: VariableSizedTrajectoryBuffer::new(env),
                device: Device::Cpu,
            };
            // TODO: other distributions/custom distributions need to be encoded
            while rollout.handle_packet::<DistributionKind>()? {}
            Ok(())
        }
        EnvConstructionMethod::DyLib => todo!(),
    }
}

#[cfg(test)]
mod test {
    use interprocess::local_socket::{
        GenericNamespaced, ListenerOptions, ToNsName, traits::ListenerExt,
    };
    use r2l_core::{
        distributions::diagonal_distribution::DiagGaussianDistribution,
        ipc::{PacketToReceive, PacketToSend, receive_packet, send_packet},
    };
    use r2l_gym::GymEnv;
    use std::io::BufReader;

    const SOCKET_NAME: &str = "test-socket";
    const ENV_NAME: &str = "Pendulum-v1";

    #[test]
    fn test_subproc() {
        let socket_name = SOCKET_NAME.to_ns_name::<GenericNamespaced>().unwrap();
        let opts = ListenerOptions::new().name(socket_name);
        let listener = opts.create_sync().unwrap();
        let child = std::process::Command::new("cargo")
            .args([
                "run",
                "--bin",
                "subproc_env",
                "--",
                "--env-construction-method",
                "gym-env",
                "--env-name",
                "Pendulum-v1",
                "--rollout-type",
                "step-bound",
                "--steps",
                "2048",
                "--socket-name",
                SOCKET_NAME,
            ])
            .spawn()
            .unwrap();
        let conn = listener.incoming().next().unwrap().unwrap();
        let mut conn = BufReader::new(conn);
        let env = GymEnv::new(ENV_NAME, None);

        send_packet(&mut conn, PacketToSend::<DiagGaussianDistribution>::Halt);
        let packet: PacketToReceive<DiagGaussianDistribution> = receive_packet(&mut conn);
        assert!(matches!(packet, PacketToReceive::Halting));
    }
}
