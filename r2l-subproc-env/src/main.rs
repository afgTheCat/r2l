// Eventually we want to load the environments as dynamic libraries via libloading or dlopen-rs
// and supported gym env as well. Also we can experiment with fork/clone whatever on linux

use candle_core::{Device, Error, Result};
use clap::Parser;
use interprocess::local_socket::{
    GenericNamespaced, Stream, ToNsName, traits::Stream as StreamTrait,
};
use r2l_core::{
    distributions::{
        Distribution, DistributionKind, diagonal_distribution::DiagGaussianDistribution,
    },
    env::{Env, RolloutMode, run_rollout},
    network::{PacketToReceive, PacketToSend, receive_packet, send_packet},
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
    rollout_type: RolloutType,

    #[arg(long)]
    steps: usize,

    #[arg(long)]
    socket_name: String,
}

pub struct Rollout<E: Env> {
    env: E,
    conn: BufReader<Stream>,
    collection_mode: RolloutMode,
}

impl<E: Env> Rollout<E> {
    fn handle_packet<D: Distribution>(&mut self) -> Result<bool> {
        let packet: PacketToReceive<D> = receive_packet(&mut self.conn);
        match packet {
            PacketToReceive::Halt => {
                send_packet(&mut self.conn, PacketToSend::<D>::Halting);
                Ok(false)
            }
            PacketToReceive::StartRollout { distribution } => {
                let rollout = run_rollout(&distribution, &self.env, self.collection_mode)?;
                let packet: PacketToSend<D> = PacketToSend::RolloutResult { rollout };
                send_packet(&mut self.conn, packet);
                Ok(true)
            }
            _ => unreachable!(),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let collection_mode = match args.rollout_type {
        RolloutType::EpisodeBound => RolloutMode::EpisodeBound {
            n_steps: args.steps,
        },
        RolloutType::StepBound => RolloutMode::StepBound {
            n_steps: args.steps,
        },
    };
    match &args.env_construction_method {
        EnvConstructionMethod::GymEnv => {
            let env = GymEnv::new(&args.env_name, None, &Device::Cpu)?;
            let socket_name = args
                .socket_name
                .to_ns_name::<GenericNamespaced>()
                .map_err(Error::wrap)?;
            let conn = Stream::connect(socket_name).unwrap();
            let conn = BufReader::new(conn);
            let mut rollout = Rollout {
                conn,
                collection_mode,
                env,
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
        network::{PacketToReceive, PacketToSend, receive_packet, send_packet},
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
        let env = GymEnv::new(ENV_NAME, None, &candle_core::Device::Cpu).unwrap();

        send_packet(&mut conn, PacketToSend::<DiagGaussianDistribution>::Halt);
        let packet: PacketToReceive<DiagGaussianDistribution> = receive_packet(&mut conn);
        assert!(matches!(packet, PacketToReceive::Halting));
    }
}
