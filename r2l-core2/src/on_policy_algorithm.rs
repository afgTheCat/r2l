use crate::{
    agent::{Agent, AgentAct, AgentObs},
    env::{Action, Actor, Observation, SnapShot},
};

macro_rules! break_on_hook_res {
    ($hook_res:expr) => {
        if $hook_res {
            break;
        }
    };
}

#[derive(Debug, Clone, Copy)]
pub enum LearningBound {
    RolloutBound {
        total_rollouts: usize,
        current_rollout: usize,
    },
    TotalStepBound {
        total_steps: usize,
        current_step: usize,
    },
}

impl LearningBound {
    pub fn total_step_bound(total_steps: usize) -> Self {
        Self::TotalStepBound {
            total_steps: total_steps,
            current_step: 0,
        }
    }

    pub fn update_current<O: Observation, A: Action>(&mut self, snapshots: &[SnapShot<O, A>]) {
        match self {
            Self::RolloutBound {
                current_rollout, ..
            } => {
                let new_rollouts: usize = snapshots.iter().fold(0, |total, step| {
                    if step.terminated || step.trancuated {
                        total + 1
                    } else {
                        total
                    }
                });
                *current_rollout += new_rollouts;
            }
            Self::TotalStepBound { total_steps, .. } => *total_steps += snapshots.len(),
        }
    }

    pub fn bound_met(&self) -> bool {
        match self {
            Self::RolloutBound {
                total_rollouts,
                current_rollout,
            } => current_rollout >= total_rollouts,
            Self::TotalStepBound {
                total_steps,
                current_step,
            } => current_step >= total_steps,
        }
    }
}

pub trait LearningHooks {
    // call it after rollouts
    fn post_rollout_hook<O: Observation, A: Action>(
        &mut self,
        snapshots: &[SnapShot<O, A>],
    ) -> bool;

    // call it after training
    fn post_training_hook(&mut self) -> bool;
}

pub struct OnPolicyAlgorithm<E: Actor, AG: Agent, H: LearningHooks> {
    pub env_pool: E,
    pub agent: AG,
    pub hooks: H,
}

// Move this elsewhere!
struct DefaultLearningHook {
    learning_bound: LearningBound,
}

impl LearningHooks for DefaultLearningHook {
    fn post_rollout_hook<O: Observation, A: Action>(
        &mut self,
        snapshots: &[SnapShot<O, A>],
    ) -> bool {
        self.learning_bound.update_current(snapshots);
        // TODO: simple println here
        false
    }

    fn post_training_hook(&mut self) -> bool {
        self.learning_bound.bound_met()
    }
}

impl<E: Actor, AG: Agent, H: LearningHooks> OnPolicyAlgorithm<E, AG, H> {
    pub fn new(env_pool: E, agent: AG, hooks: H) -> Self {
        Self {
            env_pool,
            agent,
            hooks,
        }
    }

    // TODO: conversion between Observation and Action types should be expressed more easily
    pub fn train(&mut self)
    where
        E::Obs: From<AgentObs<AG>>,
        E::Act: From<AgentAct<AG>>,
        AgentObs<AG>: From<E::Obs>,
        AgentAct<AG>: From<E::Act>,
    {
        loop {
            // rollout phase
            let distribution = self.agent.distribution();
            let rollouts = self.env_pool.collect_rollouts(distribution);
            break_on_hook_res!(self.hooks.post_rollout_hook(&rollouts));

            // learning phase
            self.agent.learn(rollouts);
            break_on_hook_res!(self.hooks.post_training_hook());
        }
    }
}
