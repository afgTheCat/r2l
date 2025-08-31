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
use std::{collections::HashMap, marker::PhantomData};

type DistrTraitObj<'a, T> = dyn Distribution<Tensor = T> + 'a;
type DistrTraitObjPtr<'a, T> = *const DistrTraitObj<'a, T>;

struct OpaqueDistrPtr<E: Env> {
    ptr: *const (),
    env_type: PhantomData<E>,
}

unsafe impl<E: Env> Send for OpaqueDistrPtr<E> {}
unsafe impl<E: Env> Sync for OpaqueDistrPtr<E> {}

impl<E: Env> OpaqueDistrPtr<E> {
    unsafe fn to_distr_trait_obj_ptr<'a>(self) -> DistrTraitObjPtr<'a, E::Tensor> {
        unsafe { &**self.ptr.cast::<DistrTraitObjPtr<'a, E::Tensor>>() }
    }

    fn from_trait_obj<'a>(ptr: DistrTraitObjPtr<'a, E::Tensor>) -> Self {
        let ptr = ptr as *const ();
        Self {
            ptr,
            env_type: PhantomData,
        }
    }
}

// type DistrPtrType = *const dyn Distribution<Tensor = Tensor>;
// pub struct DistrPtr(DistrPtrType);
//
// // TODO: write a safety comment here. Also these are just new types for a const pointer, accessing
// // them is already unsafe
// unsafe impl Send for DistrPtr {}
// unsafe impl Sync for DistrPtr {}

pub enum FixedSizeWorkerCommand<E: Env> {
    // Single step and return tranfer the buffers
    SingleStep {
        distr: OpaqueDistrPtr<E>,
    },
    // Multi step
    MultiStep {
        num_steps: usize,
        distr: OpaqueDistrPtr<E>,
    },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
    // Set buffers
    SetBuffer {
        buffer: FixedSizeStateBuffer<E>,
    },
    // Get env description
    GetEnvDescription,
}

pub enum FixedSizeWorkerResult<E: Env> {
    Step {
        buffer: FixedSizeStateBuffer<E>,
    },
    MultiStepOk,
    RolloutBuffer {
        buffer: RolloutBuffer<E::Tensor>,
    },
    SetBufferOk,
    EnvDescription {
        env_description: EnvironmentDescription<E::Tensor>,
    },
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
                FixedSizeWorkerCommand::SingleStep { distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr.to_distr_trait_obj_ptr() };
                    self.buffer.step(distr);
                    let buffer = self.buffer.move_buffer();
                    self.tx
                        .send(FixedSizeWorkerResult::Step { buffer })
                        .unwrap();
                }
                FixedSizeWorkerCommand::MultiStep { num_steps, distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr.to_distr_trait_obj_ptr() };
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

// TODO: iterating through all the channels twice is a common pattern here. Can automate?
impl<E: Env> FixedSizeEnvPool for FixedSizeThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn step<D: Distribution<Tensor = E::Tensor>>(&mut self, distr: &D, steps: usize) {
        let num_envs = self.num_envs();
        let ptr: DistrTraitObjPtr<_> = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::MultiStep {
                num_steps: steps,
                distr: OpaqueDistrPtr::from_trait_obj(ptr),
            })
            .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn step_take_buffers<D: Distribution<Tensor = E::Tensor>>(
        &mut self,
        distr: &D,
    ) -> Vec<FixedSizeStateBuffer<Self::Env>> {
        let num_envs = self.num_envs();
        let ptr: DistrTraitObjPtr<_> = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(FixedSizeWorkerCommand::SingleStep {
                distr: OpaqueDistrPtr::from_trait_obj(ptr),
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

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer<E::Tensor>> {
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
            buffs.push(buffer.into());
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

pub enum VariableSizedWorkerCommand<E: Env> {
    // Multi step
    StepMultipleWithStepBound {
        num_steps: usize,
        distr: OpaqueDistrPtr<E>,
    },
    // Return the trajectory buffer. This will probably not be needed
    ReturnRolloutBuffer,
    // Return the environment description
    GetEnvDescription,
}

pub enum VariableSizedWorkerResult<E: Env> {
    StepMultipleWithStepBoundOk,
    RolloutBuffer {
        buffer: RolloutBuffer<E::Tensor>,
    },
    EnvDescription {
        env_description: EnvironmentDescription<E::Tensor>,
    },
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
                VariableSizedWorkerCommand::StepMultipleWithStepBound { num_steps, distr } => {
                    // SAFETY: the caller guarnatees that the reference will be valid for the duration
                    // of the work. The raw pointer will not be used beyond the unsafe block
                    let distr = unsafe { &*distr.to_distr_trait_obj_ptr() };
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
            Sender<VariableSizedWorkerCommand<E>>,
            Receiver<VariableSizedWorkerResult<E>>,
        ),
    >,
    _env: PhantomData<E>,
}

impl<E: Env> VariableSizedThreadEnvPool<E> {
    pub fn new(
        channels: HashMap<
            usize,
            (
                Sender<VariableSizedWorkerCommand<E>>,
                Receiver<VariableSizedWorkerResult<E>>,
            ),
        >,
    ) -> Self {
        Self {
            channels,
            _env: PhantomData,
        }
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

impl<E: Env> VariableSizedEnvPool for VariableSizedThreadEnvPool<E> {
    type Env = E;

    fn num_envs(&self) -> usize {
        self.channels.len()
    }

    fn to_rollout_buffers(&mut self) -> Vec<RolloutBuffer<E::Tensor>> {
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

    fn step_with_episode_bound<D: Distribution<Tensor = E::Tensor>>(
        &mut self,
        distr: &D,
        steps: usize,
    ) {
        let num_envs = self.num_envs();
        let ptr: DistrTraitObjPtr<_> = distr;
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(VariableSizedWorkerCommand::StepMultipleWithStepBound {
                num_steps: steps,
                distr: OpaqueDistrPtr::from_trait_obj(ptr),
            })
            .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }
}
