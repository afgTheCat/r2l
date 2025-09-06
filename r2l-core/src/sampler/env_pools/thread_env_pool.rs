use crate::{
    distributions::Distribution,
    env::{Env, EnvironmentDescription},
    sampler::{
        env_pools::{FixedSizeEnvPool, VariableSizedEnvPool},
        trajectory_buffers::{
            fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
            variable_size_buffer::VariableSizedTrajectoryBuffer,
        },
    },
    utils::rollout_buffer::RolloutBuffer,
};
use crossbeam::channel::{Receiver, Sender};
use std::collections::HashMap;

pub enum FixedSizeWorkerCommand<E: Env> {
    // Single step and return tranfer the buffers
    SingleStep,
    // },
    MultiStep {
        num_steps: usize,
    },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
    // Set buffers
    SetBuffer {
        buffer: Box<FixedSizeStateBuffer<E>>,
    },
    // Get env description
    GetEnvDescription,
    // Set the distr
    SetDistr {
        distr: Box<dyn Distribution<Tensor = E::Tensor>>,
    },
}

pub enum FixedSizeWorkerResult<E: Env> {
    Step2 {
        buffer: Box<FixedSizeStateBuffer<E>>,
    },
    MultiStepOk,
    RolloutBuffer {
        buffer: Box<RolloutBuffer<E::Tensor>>,
    },
    SetBufferOk,
    EnvDescription {
        env_description: EnvironmentDescription<E::Tensor>,
    },
    SetDistrOk,
}

pub struct FixedSizeWorkerThread<E: Env> {
    tx: Sender<FixedSizeWorkerResult<E>>,
    rx: Receiver<FixedSizeWorkerCommand<E>>,
    buffer: FixedSizeTrajectoryBuffer<E>,
}

impl<E: Env> FixedSizeWorkerThread<E> {
    pub fn new(
        tx: Sender<FixedSizeWorkerResult<E>>,
        rx: Receiver<FixedSizeWorkerCommand<E>>,
        env: E,
        capacity: usize,
    ) -> Self {
        Self {
            tx,
            rx,
            buffer: FixedSizeTrajectoryBuffer::new(env, capacity),
        }
    }

    pub fn handle_commands(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                FixedSizeWorkerCommand::ReturnRolloutBuffer => {
                    let buffer = self.buffer.take_rollout_buffer();
                    self.tx
                        .send(FixedSizeWorkerResult::RolloutBuffer {
                            buffer: Box::new(buffer),
                        })
                        .unwrap();
                }
                FixedSizeWorkerCommand::SetBuffer { buffer } => {
                    self.buffer.set_buffer(*buffer);
                    self.tx.send(FixedSizeWorkerResult::SetBufferOk {}).unwrap();
                }
                FixedSizeWorkerCommand::GetEnvDescription => {
                    let env_description = self.buffer.env.env_description();
                    self.tx
                        .send(FixedSizeWorkerResult::EnvDescription { env_description })
                        .unwrap();
                }
                FixedSizeWorkerCommand::SetDistr { distr } => {
                    self.buffer.set_distr(Some(distr));
                    self.tx.send(FixedSizeWorkerResult::SetDistrOk).unwrap();
                }
                FixedSizeWorkerCommand::SingleStep => {
                    self.buffer.step2();
                    let buffer = self.buffer.move_buffer();
                    self.tx
                        .send(FixedSizeWorkerResult::Step2 {
                            buffer: Box::new(buffer),
                        })
                        .unwrap();
                }
                FixedSizeWorkerCommand::MultiStep { num_steps } => {
                    self.buffer.step_n2(num_steps);
                    self.tx.send(FixedSizeWorkerResult::MultiStepOk {}).unwrap();
                }
            }
        }
    }
}

pub type FixedSizeChannelPool<E> = HashMap<
    usize,
    (
        Sender<FixedSizeWorkerCommand<E>>,
        Receiver<FixedSizeWorkerResult<E>>,
    ),
>;

pub struct FixedSizeThreadEnvPool<E: Env> {
    channels: FixedSizeChannelPool<E>,
}

impl<E: Env> FixedSizeThreadEnvPool<E> {
    pub fn new(channels: FixedSizeChannelPool<E>) -> Self {
        Self { channels }
    }

    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        let (sender, receiver) = self.channels.get(&0).unwrap();
        sender
            .send(FixedSizeWorkerCommand::GetEnvDescription)
            .unwrap();
        let FixedSizeWorkerResult::EnvDescription { env_description } = receiver.recv().unwrap()
        else {
            panic!()
        };
        env_description
    }
}

impl<E: Env> FixedSizeEnvPool for FixedSizeThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn step_take_buffers(&mut self) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::SingleStep {}).unwrap();
        }
        let mut buffs = Vec::with_capacity(num_envs);
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let FixedSizeWorkerResult::Step2 { buffer } = rx.recv().unwrap() else {
                panic!()
            };
            buffs.push(*buffer);
        }
        buffs
    }

    fn set_distr<D: Distribution<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
    ) {
        for idx in 0..self.num_envs() {
            let distr: Box<dyn Distribution<Tensor = E::Tensor>> = Box::new(distr.clone());
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::SetDistr { distr }).unwrap();
        }
        for idx in 0..self.channels.len() {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn step_n<D: Distribution<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        self.set_distr(distr);
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::MultiStep { num_steps: steps })
                .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::ReturnRolloutBuffer)
                .unwrap();
        }
        let mut buffs = Vec::with_capacity(num_envs);
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            // TODO: this is ugly, implement the Into trait wherever we can
            let FixedSizeWorkerResult::RolloutBuffer { buffer } = rx.recv().unwrap() else {
                panic!()
            };
            buffs.push(*buffer);
        }
        buffs
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        for (idx, buffer) in buffers.into_iter().enumerate() {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::SetBuffer {
                buffer: Box::new(buffer),
            })
            .unwrap()
        }
        for idx in 0..self.channels.len() {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn take_rollout_buffers(&mut self) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::ReturnRolloutBuffer)
                .unwrap();
        }
        let mut buffs = Vec::with_capacity(num_envs);
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            // TODO: this is ugly, implement the Into trait wherever we can
            let FixedSizeWorkerResult::RolloutBuffer { buffer } = rx.recv().unwrap() else {
                panic!()
            };
            buffs.push(*buffer);
        }
        buffs
    }
}

pub enum VariableSizedWorkerCommand<E: Env> {
    // Multi step
    StepMultipleWithStepBound2 {
        num_steps: usize,
    },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
    // Return the environment description
    GetEnvDescription,
    // Set the distr
    SetDistr {
        distr: Box<dyn Distribution<Tensor = E::Tensor>>,
    },
}

pub enum VariableSizedWorkerResult<E: Env> {
    RolloutBuffer {
        buffer: RolloutBuffer<E::Tensor>,
    },
    EnvDescription {
        env_description: EnvironmentDescription<E::Tensor>,
    },
    SetDistrOk,
    StepMultipleWithStepBound2Ok,
}

pub struct VariableSizedWorkerThread<E: Env> {
    tx: Sender<VariableSizedWorkerResult<E>>,
    rx: Receiver<VariableSizedWorkerCommand<E>>,
    buffer: VariableSizedTrajectoryBuffer<E>,
}

impl<E: Env> VariableSizedWorkerThread<E> {
    pub fn new(
        tx: Sender<VariableSizedWorkerResult<E>>,
        rx: Receiver<VariableSizedWorkerCommand<E>>,
        env: E,
    ) -> Self {
        Self {
            tx,
            rx,
            buffer: VariableSizedTrajectoryBuffer::new(env),
        }
    }

    pub fn handle_commmand(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                VariableSizedWorkerCommand::ReturnRolloutBuffer => {
                    let buffer = self.buffer.take_rollout_buffer();
                    self.tx
                        .send(VariableSizedWorkerResult::RolloutBuffer { buffer })
                        .unwrap();
                }
                VariableSizedWorkerCommand::GetEnvDescription => {
                    let env_description = self.buffer.env.env_description();
                    self.tx
                        .send(VariableSizedWorkerResult::EnvDescription { env_description })
                        .unwrap();
                }
                VariableSizedWorkerCommand::SetDistr { distr } => {
                    self.buffer.set_distr(Some(distr));
                    self.tx.send(VariableSizedWorkerResult::SetDistrOk).unwrap();
                }
                VariableSizedWorkerCommand::StepMultipleWithStepBound2 { num_steps } => {
                    self.buffer.step_with_epiosde_bound2(num_steps);
                    self.tx
                        .send(VariableSizedWorkerResult::StepMultipleWithStepBound2Ok)
                        .unwrap();
                }
            }
        }
    }
}

type VariableSizedChannelMap<E> = HashMap<
    usize,
    (
        Sender<VariableSizedWorkerCommand<E>>,
        Receiver<VariableSizedWorkerResult<E>>,
    ),
>;

pub struct VariableSizedThreadEnvPool<E: Env> {
    channels: VariableSizedChannelMap<E>,
}

impl<E: Env> VariableSizedEnvPool for VariableSizedThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    // TODO: we should have one kind of command that sets the distribution and makes the env go
    // brrr
    fn step_with_episode_bound<D: Distribution<Tensor = <Self::Env as Env>::Tensor> + Clone>(
        &mut self,
        distr: D,
        steps: usize,
    ) -> Vec<RolloutBuffer<<Self::Env as Env>::Tensor>> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            let distr: Box<dyn Distribution<Tensor = E::Tensor>> = Box::new(distr.clone());
            tx.send(VariableSizedWorkerCommand::SetDistr { distr })
                .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }

        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(VariableSizedWorkerCommand::StepMultipleWithStepBound2 { num_steps: steps })
                .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(VariableSizedWorkerCommand::ReturnRolloutBuffer)
                .unwrap();
        }
        let mut buffs = Vec::with_capacity(num_envs);
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let VariableSizedWorkerResult::RolloutBuffer { buffer } = rx.recv().unwrap() else {
                panic!()
            };
            buffs.push(buffer);
        }
        buffs
    }
}

impl<E: Env> VariableSizedThreadEnvPool<E> {
    pub fn new(channels: VariableSizedChannelMap<E>) -> Self {
        Self { channels }
    }

    pub fn env_description(&self) -> EnvironmentDescription<E::Tensor> {
        let (tx, rx) = self.channels.get(&0).unwrap();
        tx.send(VariableSizedWorkerCommand::GetEnvDescription)
            .unwrap();
        let VariableSizedWorkerResult::EnvDescription { env_description } = rx.recv().unwrap()
        else {
            panic!()
        };
        env_description
    }
}
