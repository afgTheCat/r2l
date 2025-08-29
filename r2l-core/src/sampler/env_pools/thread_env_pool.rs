use crate::{
    distributions::Distribution,
    env::{Env, EnvironmentDescription},
    numeric::Buffer,
    sampler::{
        env_pools::{FixedSizeEnvPool, VariableSizedEnvPool},
        trajectory_buffers::{
            fixed_size_buffer::{FixedSizeStateBuffer, FixedSizeTrajectoryBuffer},
            variable_size_buffer::VariableSizedTrajectoryBuffer,
        },
    },
    utils::rollout_buffer::RolloutBuffer,
};
use candle_core::Tensor;
use crossbeam::channel::{Receiver, Sender};
use std::{collections::HashMap, marker::PhantomData};

type DistrPtrType = *const dyn Distribution<Tensor = Tensor>;
pub struct DistrPtr(DistrPtrType);

// TODO: write a safety comment here. Also these are just new types for a const pointer, accessing
// them is already unsafe
unsafe impl Send for DistrPtr {}
unsafe impl Sync for DistrPtr {}

pub enum FixedSizeWorkerCommand<E: Env> {
    // Single step and return tranfer the buffers
    SingleStep { distr: DistrPtr },
    // Multi step
    MultiStep { num_steps: usize, distr: DistrPtr },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
    // Set buffers
    SetBuffer { buffer: FixedSizeStateBuffer<E> },
    // Get env description
    GetEnvDescription,
}

pub enum FixedSizeWorkerResult<E: Env> {
    Step {
        buffer: FixedSizeStateBuffer<E>,
    },
    MultiStepOk,
    RolloutBuffer {
        buffer: RolloutBuffer<Tensor>,
    },
    SetBufferOk,
    EnvDescription {
        env_description: EnvironmentDescription,
    },
}

pub struct FixedSizeWorkerThread<E: Env<Tensor = Buffer>> {
    tx: Sender<FixedSizeWorkerResult<E>>,
    rx: Receiver<FixedSizeWorkerCommand<E>>,
    buffer: FixedSizeTrajectoryBuffer<E>,
}

impl<E: Env<Tensor = Buffer>> FixedSizeWorkerThread<E> {
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
                FixedSizeWorkerCommand::SingleStep { distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr.0 };
                    self.buffer.step(distr);
                    let buffer = self.buffer.move_buffer();
                    self.tx
                        .send(FixedSizeWorkerResult::Step { buffer })
                        .unwrap();
                }
                FixedSizeWorkerCommand::MultiStep { num_steps, distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr.0 };
                    self.buffer.step_n(distr, num_steps);
                    self.tx.send(FixedSizeWorkerResult::MultiStepOk {}).unwrap();
                }
                FixedSizeWorkerCommand::ReturnRolloutBuffer => {
                    let buffer = self.buffer.to_rollout_buffer();
                    self.tx
                        .send(FixedSizeWorkerResult::RolloutBuffer { buffer })
                        .unwrap();
                }
                FixedSizeWorkerCommand::SetBuffer { buffer } => {
                    self.buffer.set_buffer(buffer);
                    self.tx.send(FixedSizeWorkerResult::SetBufferOk {}).unwrap();
                }
                FixedSizeWorkerCommand::GetEnvDescription => {
                    let env_description = self.buffer.env.env_description();
                    self.tx
                        .send(FixedSizeWorkerResult::EnvDescription { env_description })
                        .unwrap();
                }
            }
        }
    }
}

pub struct FixedSizeThreadEnvPool<E: Env> {
    channels: HashMap<
        usize,
        (
            Sender<FixedSizeWorkerCommand<E>>,
            Receiver<FixedSizeWorkerResult<E>>,
        ),
    >,
}

impl<E: Env> FixedSizeThreadEnvPool<E> {
    pub fn new(
        channels: HashMap<
            usize,
            (
                Sender<FixedSizeWorkerCommand<E>>,
                Receiver<FixedSizeWorkerResult<E>>,
            ),
        >,
    ) -> Self {
        Self { channels }
    }

    pub fn env_description(&self) -> EnvironmentDescription {
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

// TODO: iterating through all the channels twice is a common pattern here. Can automate?
impl<E: Env<Tensor = Buffer>> FixedSizeEnvPool for FixedSizeThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn step<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        let num_envs = self.num_envs();
        let ptr: DistrPtrType = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::MultiStep {
                num_steps: steps,
                distr: DistrPtr(ptr),
            })
            .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn step_take_buffers<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        let num_envs = self.num_envs();
        let ptr: DistrPtrType = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::SingleStep {
                distr: DistrPtr(ptr),
            })
            .unwrap();
        }
        let mut buffs = Vec::with_capacity(num_envs);
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let FixedSizeWorkerResult::Step { buffer } = rx.recv().unwrap() else {
                panic!()
            };
            buffs.push(buffer);
        }
        buffs
    }

    fn to_rollout_buffers(&self) -> Vec<RolloutBuffer<Tensor>> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::ReturnRolloutBuffer)
                .unwrap();
        }
        let mut buffs = Vec::with_capacity(num_envs);
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let FixedSizeWorkerResult::RolloutBuffer { buffer } = rx.recv().unwrap() else {
                panic!()
            };
            buffs.push(buffer);
        }
        buffs
    }

    fn set_buffers(&mut self, buffers: Vec<FixedSizeStateBuffer<Self::Env>>) {
        for (idx, buffer) in buffers.into_iter().enumerate() {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::SetBuffer { buffer })
                .unwrap()
        }
        for idx in 0..self.channels.len() {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }
}

pub enum VariableSizedWorkerCommand {
    // Multi step
    StepMultipleWithStepBound { num_steps: usize, distr: DistrPtr },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
    // Return the environment description
    GetEnvDescription,
}

pub enum VariableSizedWorkerResult {
    StepMultipleWithStepBoundOk,
    RolloutBuffer {
        buffer: RolloutBuffer<Tensor>,
    },
    EnvDescription {
        env_description: EnvironmentDescription,
    },
}

pub struct VariableSizedWorkerThread<E: Env<Tensor = Buffer>> {
    tx: Sender<VariableSizedWorkerResult>,
    rx: Receiver<VariableSizedWorkerCommand>,
    buffer: VariableSizedTrajectoryBuffer<E>,
}

impl<E: Env<Tensor = Buffer>> VariableSizedWorkerThread<E> {
    pub fn new(
        tx: Sender<VariableSizedWorkerResult>,
        rx: Receiver<VariableSizedWorkerCommand>,
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
                VariableSizedWorkerCommand::StepMultipleWithStepBound { num_steps, distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr.0 };
                    self.buffer.step_with_epiosde_bound(distr, num_steps);
                    self.tx
                        .send(VariableSizedWorkerResult::StepMultipleWithStepBoundOk)
                        .unwrap();
                }
                VariableSizedWorkerCommand::ReturnRolloutBuffer => {
                    let buffer = self.buffer.to_rollout_buffer();
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
            }
        }
    }
}

pub struct VariableSizedThreadEnvPool<E: Env> {
    channels: HashMap<
        usize,
        (
            Sender<VariableSizedWorkerCommand>,
            Receiver<VariableSizedWorkerResult>,
        ),
    >,
    _env: PhantomData<E>,
}

impl<E: Env> VariableSizedThreadEnvPool<E> {
    pub fn new(
        channels: HashMap<
            usize,
            (
                Sender<VariableSizedWorkerCommand>,
                Receiver<VariableSizedWorkerResult>,
            ),
        >,
    ) -> Self {
        Self {
            channels,
            _env: PhantomData,
        }
    }

    pub fn env_description(&self) -> EnvironmentDescription {
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

impl<E: Env<Tensor = Buffer>> VariableSizedEnvPool for VariableSizedThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer<Tensor>> {
        let num_envs = self.num_envs();
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

    fn step_with_episode_bound<D: Distribution<Tensor = Tensor>>(
        &mut self,
        distr: &D,
        steps: usize,
    ) {
        let num_envs = self.num_envs();
        let ptr: DistrPtrType = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(VariableSizedWorkerCommand::StepMultipleWithStepBound {
                num_steps: steps,
                distr: DistrPtr(ptr),
            })
            .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }
}
