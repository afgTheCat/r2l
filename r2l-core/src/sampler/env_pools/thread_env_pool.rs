use crate::{
    distributions::Distribution,
    env::Env,
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

type DistrPtr = *const dyn Distribution<Tensor = Tensor>;

enum FixedSizeWorkerCommand<E: Env> {
    // Single step and return tranfer the buffers
    SingleStep { distr: DistrPtr },
    // Multi step
    MultiStep { num_steps: usize, distr: DistrPtr },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
    // Set buffers
    SetBuffer { buffer: FixedSizeStateBuffer<E> },
}

enum FixedSizeWorkerResult<E: Env> {
    Step { buffer: FixedSizeStateBuffer<E> },
    MultiStepOk {},
    RolloutBuffer { buffer: RolloutBuffer },
    SetBufferOk {},
}

pub struct FixedSizeWorkerThread<E: Env<Tensor = Buffer>> {
    tx: Sender<FixedSizeWorkerResult<E>>,
    rx: Receiver<FixedSizeWorkerCommand<E>>,
    buffer: FixedSizeTrajectoryBuffer<E>,
}

impl<E: Env<Tensor = Buffer>> FixedSizeWorkerThread<E> {
    pub fn handle_commmand(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                FixedSizeWorkerCommand::SingleStep { distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr };
                    self.buffer.step(distr);
                    let buffer = self.buffer.move_buffer();
                    self.tx
                        .send(FixedSizeWorkerResult::Step { buffer })
                        .unwrap();
                }
                FixedSizeWorkerCommand::MultiStep { num_steps, distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr };
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

// TODO: iterating through all the channels twice is a common pattern here. Can automate?
impl<E: Env<Tensor = Buffer>> FixedSizeEnvPool for FixedSizeThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn step<D: Distribution<Tensor = Tensor>>(&mut self, distr: &D, steps: usize) {
        let num_envs = self.num_envs();
        let ptr: DistrPtr = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::MultiStep {
                num_steps: steps,
                distr: ptr,
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
        let ptr: DistrPtr = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::SingleStep { distr: ptr })
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

    fn to_rollout_buffers(&self) -> Vec<RolloutBuffer> {
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

enum VariableSizedWorkerCommand {
    // Multi step
    StepMultipleWithStepBound { num_steps: usize, distr: DistrPtr },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
}

enum VariableSizedWorkerResult {
    StepMultipleWithStepBoundOk,
    RolloutBuffer { buffer: RolloutBuffer },
}

pub struct VariableSizedWorkerThread<E: Env<Tensor = Buffer>> {
    tx: Sender<VariableSizedWorkerResult>,
    rx: Receiver<VariableSizedWorkerCommand>,
    buffer: VariableSizedTrajectoryBuffer<E>,
}

impl<E: Env<Tensor = Buffer>> VariableSizedWorkerThread<E> {
    pub fn handle_commmand(&mut self) {
        loop {
            let command = self.rx.recv().unwrap();
            match command {
                VariableSizedWorkerCommand::StepMultipleWithStepBound { num_steps, distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr };
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

impl<E: Env<Tensor = Buffer>> VariableSizedEnvPool for VariableSizedThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer> {
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
        let ptr: DistrPtr = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(VariableSizedWorkerCommand::StepMultipleWithStepBound {
                num_steps: steps,
                distr: ptr,
            })
            .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }
}
