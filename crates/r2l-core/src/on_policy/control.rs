use std::path::PathBuf;

use crossbeam::channel::{Receiver, Sender, unbounded};

use crate::models::Actor;

enum OnPolCmd {
    Shutdown,
    SerializePolicy(PathBuf),
}

/// Handle for controlling a running on-policy algorithm.
#[derive(Clone)]
pub struct OnPolControl {
    sender: Sender<OnPolCmd>,
}

/// Receiving side owned by the on-policy training loop.
#[doc(hidden)]
pub struct OnPolCommands {
    receiver: Receiver<OnPolCmd>,
}

impl OnPolControl {
    #[doc(hidden)]
    pub fn channel() -> (Self, OnPolCommands) {
        let (sender, receiver) = unbounded();
        (Self { sender }, OnPolCommands { receiver })
    }

    /// Requests a clean stop after the current update.
    pub fn shutdown(&self) {
        self.sender.send(OnPolCmd::Shutdown).unwrap();
    }

    /// Requests serialization of the current actor after the current update.
    pub fn serialize_policy<P: Into<PathBuf>>(&self, path: P) {
        self.sender
            .send(OnPolCmd::SerializePolicy(path.into()))
            .unwrap();
    }
}

impl OnPolCommands {
    pub fn process<A: Actor>(&self, actor: impl FnOnce() -> A) -> bool {
        let mut stop = false;
        let mut paths = Vec::new();
        for command in self.receiver.try_iter() {
            match command {
                OnPolCmd::Shutdown => stop = true,
                OnPolCmd::SerializePolicy(path) if !paths.contains(&path) => paths.push(path),
                OnPolCmd::SerializePolicy(_) => {}
            }
        }

        if !paths.is_empty() {
            let bytes = actor()
                .try_serialize()
                .expect("actor does not support serialization");
            for path in paths {
                std::fs::write(path, &bytes).expect("failed to write actor");
            }
        }
        stop
    }
}
