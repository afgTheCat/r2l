use std::thread::JoinHandle;

use crossbeam::channel::{Receiver, Sender, unbounded};
use r2l_core::{
    error::{Error, ResourceInterrupted, Result},
    models::Actor,
    rng::{sample_u64, set_seed},
    tensor::R2lTensor,
};

use crate::SamplerExecutionMode;

pub(super) trait PoolWorker {
    type Tensor: R2lTensor;
    type Request: Copy + Send + 'static;
    type Output: Send + 'static;

    fn set_policy(&mut self, actor: Box<dyn Actor<Tensor = Self::Tensor>>);
    fn collect(&mut self, request: Self::Request) -> Result<Self::Output>;
    fn reset(&mut self) -> Result<()>;
}

enum Command<T: R2lTensor, R> {
    SetPolicy(Box<dyn Actor<Tensor = T>>),
    Collect(R),
    Reset,
    Shutdown,
}

enum Response<O> {
    Done(Result<()>),
    Collected(Result<O>),
}

impl<O> Response<O> {
    fn into_done(self) -> Result<()> {
        match self {
            Self::Done(result) => result,
            Self::Collected(_) => Err(Self::interrupted("unexpected collection response")),
        }
    }

    fn into_output(self) -> Result<O> {
        match self {
            Self::Collected(result) => result,
            Self::Done(_) => Err(Self::interrupted("expected a collection response")),
        }
    }

    fn interrupted(details: &str) -> Error {
        ResourceInterrupted {
            resource: "sampler worker".into(),
            details: details.into(),
        }
        .into()
    }
}

pub(super) struct ThreadWorker<W: PoolWorker> {
    commands: Sender<Command<W::Tensor, W::Request>>,
    responses: Receiver<Response<W::Output>>,
    handle: Option<JoinHandle<()>>,
}

impl<W: PoolWorker + 'static> ThreadWorker<W> {
    fn build(factory: impl FnOnce() -> Result<W> + Send + 'static) -> Result<Self> {
        let worker = Self::spawn(factory);
        worker.wait_ready()?;
        Ok(worker)
    }

    pub(super) fn spawn(factory: impl FnOnce() -> Result<W> + Send + 'static) -> Self {
        let (commands, command_rx) = unbounded();
        let (response_tx, responses) = unbounded();
        let seed = sample_u64();
        let handle = std::thread::spawn(move || {
            set_seed(seed);
            let mut worker = match factory() {
                Ok(worker) => worker,
                Err(error) => {
                    let _ = response_tx.send(Response::Done(Err(error)));
                    return;
                }
            };
            if response_tx.send(Response::Done(Ok(()))).is_err() {
                return;
            }
            while let Ok(command) = command_rx.recv() {
                let response = match command {
                    Command::SetPolicy(actor) => {
                        worker.set_policy(actor);
                        Response::Done(Ok(()))
                    }
                    Command::Collect(request) => Response::Collected(worker.collect(request)),
                    Command::Reset => Response::Done(worker.reset()),
                    Command::Shutdown => break,
                };
                if response_tx.send(response).is_err() {
                    break;
                }
            }
        });
        Self {
            commands,
            responses,
            handle: Some(handle),
        }
    }

    pub(super) fn wait_ready(&self) -> Result<()> {
        self.receive()?.into_done()
    }

    fn receive(&self) -> Result<Response<W::Output>> {
        self.responses
            .recv()
            .map_err(|_| Response::<W::Output>::interrupted("result channel disconnected"))
    }
}

impl<W: PoolWorker> Drop for ThreadWorker<W> {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(super) enum Workers<W: PoolWorker> {
    Inline(Vec<W>),
    Threaded(Vec<ThreadWorker<W>>),
}

impl<W: PoolWorker + 'static> Workers<W> {
    pub(super) fn build(
        factories: impl Iterator<Item = impl FnOnce() -> Result<W> + Send + 'static>,
        execution_mode: SamplerExecutionMode,
    ) -> Result<Self> {
        match execution_mode {
            SamplerExecutionMode::SingleThreaded => factories
                .map(|factory| factory())
                .collect::<Result<_>>()
                .map(Self::Inline),
            SamplerExecutionMode::MultiThreaded => factories
                .map(ThreadWorker::build)
                .collect::<Result<_>>()
                .map(Self::Threaded),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Inline(workers) => workers.len(),
            Self::Threaded(workers) => workers.len(),
        }
    }

    // Drain every dispatched command before returning an error, so no worker is
    // still holding a state/buffer lock and the next request sees fresh replies.
    fn dispatch(
        workers: &[ThreadWorker<W>],
        indices: impl Iterator<Item = usize> + Clone,
        mut command: impl FnMut() -> Command<W::Tensor, W::Request>,
        mut on_response: impl FnMut(usize, Response<W::Output>) -> Result<()>,
    ) -> Result<()> {
        let mut result = Ok(());
        for idx in indices.clone() {
            result =
                result.and(workers[idx].commands.send(command()).map_err(|_| {
                    Response::<W::Output>::interrupted("command channel disconnected")
                }));
        }
        for idx in indices {
            // A failed send means the worker is exiting; its response channel
            // will also close. Always receive, even after an earlier error.
            let response = workers[idx]
                .receive()
                .and_then(|response| on_response(idx, response));
            result = result.and(response);
        }
        result
    }

    pub(super) fn set_policy<A: Actor<Tensor = W::Tensor> + Clone>(
        &mut self,
        actor: &A,
    ) -> Result<()> {
        match self {
            Self::Inline(workers) => {
                for worker in workers {
                    worker.set_policy(Box::new(actor.clone()));
                }
                Ok(())
            }
            Self::Threaded(workers) => Self::dispatch(
                workers,
                0..workers.len(),
                || Command::SetPolicy(Box::new(actor.clone())),
                |_, response| response.into_done(),
            ),
        }
    }

    pub(super) fn collect_into(
        &mut self,
        indices: impl Iterator<Item = usize> + Clone,
        request: W::Request,
        mut output: impl FnMut(usize, W::Output),
    ) -> Result<()> {
        match self {
            Self::Inline(workers) => {
                for idx in indices {
                    output(idx, workers[idx].collect(request)?);
                }
                Ok(())
            }
            Self::Threaded(workers) => Self::dispatch(
                workers,
                indices,
                || Command::Collect(request),
                |idx, response| {
                    output(idx, response.into_output()?);
                    Ok(())
                },
            ),
        }
    }

    pub(super) fn reset(&mut self) -> Result<()> {
        match self {
            Self::Inline(workers) => workers.iter_mut().try_for_each(PoolWorker::reset),
            Self::Threaded(workers) => Self::dispatch(
                workers,
                0..workers.len(),
                || Command::Reset,
                |_, response| response.into_done(),
            ),
        }
    }
}
