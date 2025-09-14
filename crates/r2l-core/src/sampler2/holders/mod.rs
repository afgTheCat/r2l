use crate::{
    distributions::Policy,
    env::Env,
    sampler2::{Buffer, CollectionBound, EnvPool},
};
use crossbeam::channel::{Receiver, Sender};
use std::collections::HashMap;

pub enum WorkerCommand<E: Env> {
    SetPolicy(Box<dyn Policy<Tensor = E::Tensor>>),
    Collect(CollectionBound),
}

// what we can do, is have a buffer be on heap that both the worker and the main thread can reach
enum WorkerResult<B: Buffer> {
    PolicySet,
    Collected(B),
}

// TODO: better name here
pub struct WorkerThread<E: Env, B: Buffer<E = E> + Send + Clone> {
    tx: Sender<WorkerResult<B>>,
    rx: Receiver<WorkerCommand<E>>,
    buffer: B,
    env: E,
    distr: Option<Box<dyn Policy<Tensor = E::Tensor>>>,
    last_state: Option<E::Tensor>,
}

impl<E: Env, B: Buffer<E = E> + Send + Clone> WorkerThread<E, B> {
    pub fn step(&mut self) {
        let Some(distr) = &mut self.distr else {
            todo!()
        };
        let last_state = self.last_state.take();
        self.buffer.step(&mut self.env, distr, last_state);
    }

    pub fn collect(&mut self, bound: CollectionBound) {
        match bound {
            CollectionBound::StepBound { steps } => {
                for _ in 0..steps {
                    self.step();
                }
            }
            CollectionBound::EpisodeBound { steps } => {
                let mut steps_taken = 0;
                loop {
                    self.step();
                    steps_taken += 1;
                    if steps_taken >= steps && self.buffer.last_state_terminates() {
                        break;
                    }
                }
            }
        }
    }

    pub fn handle_commands(&mut self) {
        let command = self.rx.recv().unwrap();
        match command {
            WorkerCommand::SetPolicy(distr) => {
                self.distr = Some(distr);
                self.tx.send(WorkerResult::PolicySet).unwrap();
            }
            WorkerCommand::Collect(bound) => {
                self.collect(bound);
                self.tx
                    .send(WorkerResult::Collected(self.buffer.clone()))
                    .unwrap();
            }
        }
    }
}

type ChannelHashMap<E, B> = HashMap<usize, (Sender<WorkerCommand<E>>, Receiver<WorkerResult<B>>)>;

pub struct WorkerPool<E: Env, B: Buffer<E = E>> {
    collection_bound: CollectionBound,
    channels: ChannelHashMap<E, B>,
}

impl<E: Env, B: Buffer<E = E>> WorkerPool<E, B> {
    fn num_envs(&self) -> usize {
        self.channels.len()
    }
}

impl<E: Env, B: Buffer<E = E> + Send + Clone> EnvPool for WorkerPool<E, B> {
    type E = E;
    type B = B;

    fn collection_bound(&self) -> CollectionBound {
        self.collection_bound.clone()
    }

    fn set_policy<P: Policy<Tensor = <Self::E as Env>::Tensor> + Clone>(&mut self, policy: P) {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::SetPolicy(Box::new(policy.clone())))
                .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn get_buffers(&self) -> Vec<Self::B> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::Collect(CollectionBound::StepBound {
                steps: 0,
            }))
            .unwrap();
        }
        let mut buffers = vec![];
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let WorkerResult::Collected(buffer) = rx.recv().unwrap() else {
                panic!()
            };
            buffers.push(buffer);
        }
        buffers
    }

    fn single_step(&mut self) {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::Collect(CollectionBound::StepBound {
                steps: 1,
            }))
            .unwrap();
        }
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            rx.recv().unwrap();
        }
    }

    fn collect(&self) -> Vec<Self::B> {
        let num_envs = self.num_envs();
        for idx in 0..num_envs {
            let tx = &self.channels.get(&idx).unwrap().0;
            tx.send(WorkerCommand::Collect(self.collection_bound.clone()))
                .unwrap();
        }
        let mut buffers = vec![];
        for idx in 0..num_envs {
            let rx = &self.channels.get(&idx).unwrap().1;
            let WorkerResult::Collected(buffer) = rx.recv().unwrap() else {
                panic!()
            };
            buffers.push(buffer);
        }
        buffers
    }
}
