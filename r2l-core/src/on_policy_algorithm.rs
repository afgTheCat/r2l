use crate::Algorithm;
use crate::agents::Agent2;
use crate::env::{EnvPool, RolloutMode};
use crate::utils::rollout_buffer::RolloutBuffer;
use candle_core::Result;
use r2l_macros::training_hook;

#[derive(Debug, Clone, Copy)]
pub enum LearningSchedule {
    RolloutBound {
        total_rollouts: usize,
        current_rollout: usize,
    },
    TotalStepBound {
        total_steps: usize,
        current_step: usize,
    },
}

impl LearningSchedule {
    pub fn total_step_bound(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps: total_steps,
            current_step: 0,
        }
    }
}

#[training_hook]
pub trait BeforeTrainingHook {
    fn call_hook(&mut self) -> candle_core::Result<bool>;
}

#[training_hook]
#[allow(clippy::ptr_arg)]
pub trait TrainingHook {
    fn call_hook(
        &mut self,
        learning_schedule: &LearningSchedule,
        rollouts: &Vec<RolloutBuffer>,
    ) -> candle_core::Result<bool>;
}

#[training_hook]
pub trait AfterTrainingHook {
    fn call_hook(&mut self) -> candle_core::Result<bool>;
}

#[derive(Default)]
pub struct OnPolicyHooks {
    before_training_hook: Option<Box<dyn BeforeTrainingHook>>,
    training_hook: Option<Box<dyn TrainingHook>>,
    after_training_hook: Option<Box<dyn AfterTrainingHook>>,
}

impl OnPolicyHooks {
    pub fn add_training_hook<H>(&mut self, training_hook: impl IntoTrainingHook<H>) {
        self.training_hook = Some(training_hook.into_boxed())
    }

    fn call_before_training_hook(&mut self) -> Result<bool> {
        if let Some(hook) = &mut self.before_training_hook {
            hook.call_hook()
        } else {
            Ok(false)
        }
    }

    fn call_training_hook(
        &mut self,
        learning_schedule: &LearningSchedule,
        rollouts: &Vec<RolloutBuffer>,
    ) -> Result<bool> {
        if let Some(hook) = &mut self.training_hook {
            hook.call_hook(learning_schedule, rollouts)
        } else {
            Ok(false)
        }
    }

    fn call_after_training_hook(&mut self) -> Result<bool> {
        if let Some(hook) = &mut self.after_training_hook {
            hook.call_hook()
        } else {
            Ok(false)
        }
    }
}

pub struct OnPolicyAlgorithm2<E: EnvPool, A: Agent2> {
    pub env_pool: E,
    pub agent: A,
    pub learning_schedule: LearningSchedule,
    pub rollout_mode: RolloutMode,
    pub hooks: OnPolicyHooks,
}

impl<E: EnvPool, A: Agent2> OnPolicyAlgorithm2<E, A> {
    pub fn new(
        env_pool: E,
        agent: A,
        learning_schedule: LearningSchedule,
        rollout_mode: RolloutMode,
        hooks: OnPolicyHooks,
    ) -> Self {
        Self {
            env_pool,
            agent,
            learning_schedule,
            rollout_mode,
            hooks,
        }
    }

    fn collect_rollout(&mut self) -> Result<Option<Vec<RolloutBuffer>>> {
        let distr = self.agent.distribution();
        match &mut self.learning_schedule {
            LearningSchedule::TotalStepBound {
                total_steps,
                current_step,
            } => {
                let remaining_steps = *total_steps as isize - *current_step as isize;
                if remaining_steps <= 0 {
                    Ok(None)
                } else {
                    let rollouts = self.env_pool.collect_rollouts(distr, self.rollout_mode)?;
                    let rollout_steps: usize = rollouts.iter().map(|e| e.actions.len()).sum();
                    *total_steps -= rollout_steps;
                    Ok(Some(rollouts))
                }
            }
            LearningSchedule::RolloutBound {
                total_rollouts,
                current_rollout,
            } => {
                if current_rollout == total_rollouts {
                    Ok(None)
                } else {
                    let rollouts = self.env_pool.collect_rollouts(distr, self.rollout_mode)?;
                    *current_rollout += 1;
                    Ok(Some(rollouts))
                }
            }
        }
    }

    pub fn num_env(&self) -> usize {
        self.env_pool.num_env()
    }
}

impl<E: EnvPool, A: Agent2> Algorithm for OnPolicyAlgorithm2<E, A> {
    fn train(&mut self) -> Result<()> {
        if self.hooks.call_before_training_hook()? {
            return Ok(());
        }
        while let Some(rollouts) = self.collect_rollout()? {
            if self
                .hooks
                .call_training_hook(&self.learning_schedule, &rollouts)?
            {
                break;
            }
            self.agent.learn(rollouts)?;
        }
        self.hooks.call_after_training_hook()?;
        Ok(())
    }
}
