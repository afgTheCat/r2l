use std::{num::NonZeroUsize, sync::mpsc::Sender};

use r2l_agents::on_policy_algorithms::{a2c::A2CParams, ppo::PPOParams};

use crate::{
    A2CRolloutStats, A2CSettings, ClipRangeSchedule, PPORolloutStats, PPOSettings,
    hooks::learning::{TargetKl, reporter::RolloutReporter},
};

pub(crate) enum AlgoConfig {
    PPO {
        gamma: f32,
        lambda: f32,
        sample_size: NonZeroUsize,
        total_epochs: NonZeroUsize,
        clip_range_schedule: ClipRangeSchedule,
        target_kl: Option<f32>,
        log_progress: bool,
    },
    A2C {
        gamma: f32,
        lambda: f32,
        sample_size: NonZeroUsize,
        log_progress: bool,
    },
}

impl AlgoConfig {
    pub(crate) fn ppo_parts(
        &self,
        reporter: Option<Sender<PPORolloutStats>>,
        n_envs: NonZeroUsize,
    ) -> (PPOParams, PPOSettings) {
        let Self::PPO {
            gamma,
            lambda,
            sample_size,
            total_epochs,
            clip_range_schedule,
            target_kl,
            log_progress,
        } = self
        else {
            unreachable!("PPO agent type must use PPO configuration")
        };
        let params = PPOParams {
            clip_range: clip_range_schedule.initial_value(),
            gamma: *gamma,
            lambda: *lambda,
            sample_size: sample_size.get(),
        };
        let settings = PPOSettings {
            total_epochs: *total_epochs,
            current_epoch: 0,
            clip_range_schedule: *clip_range_schedule,
            target_kl: target_kl.map(|target| TargetKl {
                target,
                target_exceeded: false,
            }),
            reporter: RolloutReporter::new(reporter, *log_progress, n_envs),
        };
        (params, settings)
    }

    pub(crate) fn a2c_parts(
        &self,
        reporter: Option<Sender<A2CRolloutStats>>,
        n_envs: NonZeroUsize,
    ) -> (A2CParams, A2CSettings) {
        let Self::A2C {
            gamma,
            lambda,
            sample_size,
            log_progress,
        } = self
        else {
            unreachable!("A2C agent type must use A2C configuration")
        };
        let params = A2CParams {
            gamma: *gamma,
            lambda: *lambda,
            sample_size: sample_size.get(),
        };
        let settings = A2CSettings {
            reporter: RolloutReporter::new(reporter, *log_progress, n_envs),
        };
        (params, settings)
    }
}
