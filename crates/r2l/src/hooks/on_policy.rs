use std::sync::mpsc::{Receiver, Sender};

use r2l_core::error::{Error, ResourceInterrupted};

/// Commands processed by the default on-policy hooks at training boundaries.
pub enum OnPolicyCommand {
    /// Stops training before the next learning phase or after the current one.
    Shutdown,
    /// Serializes the current runtime actor to the given path.
    SerializeCurrentPolicy(String),
}

/// Acknowledgements sent after an on-policy command has been processed.
pub enum OnPolicyCommandResult {
    /// Training is stopping and runtime cleanup will follow.
    Stopping,
    /// Training stopped completely and runtime cleanup has happened.
    Stopped,
    /// Result of attempting to serialize the current runtime actor.
    CurrentPolicySerialized(Result<(), Error>),
}

/// Algorithm-side endpoint for receiving on-policy commands.
pub struct OnPolicyCommandReceiver {
    /// Receives commands from the user-side endpoint.
    pub rx: Receiver<OnPolicyCommand>,
    /// Sends command results to the user-side endpoint.
    pub tx: Sender<OnPolicyCommandResult>,
}

impl OnPolicyCommandReceiver {
    /// Creates an algorithm-side endpoint from its command and result channels.
    #[must_use]
    pub fn new(rx: Receiver<OnPolicyCommand>, tx: Sender<OnPolicyCommandResult>) -> Self {
        Self { rx, tx }
    }
}

/// User-side endpoint for sending commands to an on-policy training loop.
#[derive(Debug)]
pub struct OnPolicyCommandSender {
    /// Receives command results from the training loop.
    pub rx: Receiver<OnPolicyCommandResult>,
    /// Sends commands to the training loop.
    pub tx: Sender<OnPolicyCommand>,
}

impl OnPolicyCommandSender {
    /// Creates a user-side endpoint from its result and command channels.
    #[must_use]
    pub fn new(rx: Receiver<OnPolicyCommandResult>, tx: Sender<OnPolicyCommand>) -> Self {
        Self { rx, tx }
    }

    /// Shuts down the on-policy algorithm gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if the training-side command receiver has disconnected.
    pub fn shutdown(&self) -> Result<(), Error> {
        self.tx.send(OnPolicyCommand::Shutdown).map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy command channel".into(),
                details: error.to_string(),
            })
        })?;
        while self.rx.recv().is_ok() {}
        Ok(())
    }
}

/// Creates the algorithm-side receiver and user-side sender for on-policy commands.
#[must_use]
pub fn on_policy_command_channel() -> (OnPolicyCommandReceiver, OnPolicyCommandSender) {
    let (command_tx, command_rx) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();
    (
        OnPolicyCommandReceiver::new(command_rx, result_tx),
        OnPolicyCommandSender::new(result_rx, command_tx),
    )
}
