use crate::{distributions::Distribution, env::RolloutMode, utils::rollout_buffer::RolloutBuffer};
use bincode::{Decode, Encode};
use candle_core::Tensor;
use interprocess::local_socket::Stream;
use std::io::{BufReader, Read, Write};

#[derive(Debug, Encode)]
pub enum PacketToSend<'a, D: Distribution<Tensor = Tensor>> {
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
        rollout: RolloutBuffer<Tensor>,
    },
}

#[derive(Debug, Decode)]
pub enum PacketToReceive<D: Distribution<Tensor = Tensor>> {
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
        rollout: RolloutBuffer<Tensor>,
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
