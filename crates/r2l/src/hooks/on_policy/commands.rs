use std::{
    path::PathBuf,
    sync::mpsc::{Receiver, Sender, channel},
};

use r2l_core::{
    HookResult,
    error::{Error, ResourceInterrupted},
    models::ToSafetensors,
    on_policy::algorithm::{Agent, OnPolicyRuntime, Sampler},
};

enum OnPolicyCommand {
    StopTraining,
    SerializeCurrentPolicy(PathBuf),
}

enum OnPolicyCommandResult {
    Stopping,
    Stopped,
    CurrentPolicySerialized(Result<(), Error>),
}

/// Algorithm-side endpoint of an on-policy control channel.
pub(crate) struct OnPolicyControlEndpoint {
    /// Receives commands from the control handle.
    rx: Receiver<OnPolicyCommand>,
    /// Sends command results to the control handle.
    tx: Sender<OnPolicyCommandResult>,
}

impl OnPolicyControlEndpoint {
    /// Creates an algorithm-side control endpoint from its command and result channels.
    #[must_use]
    fn new(rx: Receiver<OnPolicyCommand>, tx: Sender<OnPolicyCommandResult>) -> Self {
        Self { rx, tx }
    }
}

/// Handle for controlling a running on-policy training loop.
#[derive(Debug)]
pub struct OnPolicyControlHandle {
    rx: Receiver<OnPolicyCommandResult>,
    tx: Sender<OnPolicyCommand>,
}

impl OnPolicyControlHandle {
    /// Creates a control handle from its result and command channels.
    #[must_use]
    fn new(rx: Receiver<OnPolicyCommandResult>, tx: Sender<OnPolicyCommand>) -> Self {
        Self { rx, tx }
    }

    fn send(&self, command: OnPolicyCommand) -> Result<(), Error> {
        self.tx.send(command).map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy control channel".into(),
                details: error.to_string(),
            })
        })
    }

    fn receive(&self) -> Result<OnPolicyCommandResult, Error> {
        self.rx.recv().map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy control channel".into(),
                details: error.to_string(),
            })
        })
    }

    /// Requests serialization of the current policy and waits for the result.
    ///
    /// # Arguments
    ///
    /// * `path` - Destination path for the serialized policy.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails, the training loop is stopping,
    /// or the training-side control endpoint disconnects.
    pub fn serialize_current_policy(&self, path: impl Into<PathBuf>) -> Result<(), Error> {
        self.send(OnPolicyCommand::SerializeCurrentPolicy(path.into()))?;
        match self.receive()? {
            OnPolicyCommandResult::CurrentPolicySerialized(result) => result,
            OnPolicyCommandResult::Stopping | OnPolicyCommandResult::Stopped => {
                Err(Error::InvalidState {
                    operation: "serialize current policy".into(),
                    details: "the training loop is stopping".into(),
                })
            }
        }
    }

    /// Requests that the current training loop stop and waits for it to finish.
    ///
    /// # Errors
    ///
    /// Returns an error if the training-side control endpoint has disconnected.
    pub fn stop_training(&self) -> Result<(), Error> {
        self.send(OnPolicyCommand::StopTraining)?;
        loop {
            if matches!(self.receive()?, OnPolicyCommandResult::Stopped) {
                return Ok(());
            }
        }
    }
}

/// Creates paired training and caller endpoints for on-policy control.
pub(crate) fn on_policy_control_channel() -> (OnPolicyControlEndpoint, OnPolicyControlHandle) {
    let (command_tx, command_rx) = channel();
    let (result_tx, result_rx) = channel();
    (
        OnPolicyControlEndpoint::new(command_rx, result_tx),
        OnPolicyControlHandle::new(result_rx, command_tx),
    )
}

pub(crate) struct OnPolicyCommandHandler {
    endpoint: Option<OnPolicyControlEndpoint>,
}

impl OnPolicyCommandHandler {
    pub(crate) fn new(endpoint: Option<OnPolicyControlEndpoint>) -> Self {
        Self { endpoint }
    }

    fn send_result(
        endpoint: &OnPolicyControlEndpoint,
        result: OnPolicyCommandResult,
    ) -> Result<(), Error> {
        endpoint.tx.send(result).map_err(|error| {
            Error::ResourceInterrupted(ResourceInterrupted {
                resource: "on-policy command result channel".into(),
                details: error.to_string(),
            })
        })
    }

    pub(super) fn process_pending<A: Agent<Actor: ToSafetensors>, S: Sampler>(
        &self,
        runtime: &mut OnPolicyRuntime<A, S>,
    ) -> Result<HookResult, Error> {
        let Some(endpoint) = &self.endpoint else {
            return Ok(HookResult::Continue);
        };
        while let Ok(command) = endpoint.rx.try_recv() {
            match command {
                OnPolicyCommand::StopTraining => {
                    Self::send_result(endpoint, OnPolicyCommandResult::Stopping)?;
                    return Ok(HookResult::Break);
                }
                OnPolicyCommand::SerializeCurrentPolicy(path) => {
                    let bytes = runtime.actor().to_safetensors()?;
                    let result = std::fs::write(path, bytes).map_err(Error::wrap);
                    Self::send_result(
                        endpoint,
                        OnPolicyCommandResult::CurrentPolicySerialized(result),
                    )?;
                }
            }
        }
        Ok(HookResult::Continue)
    }

    pub(super) fn notify_stopped(&self) -> Result<(), Error> {
        if let Some(endpoint) = &self.endpoint {
            Self::send_result(endpoint, OnPolicyCommandResult::Stopped)
        } else {
            Ok(())
        }
    }
}
