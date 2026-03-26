use anyhow::Result;
use candle_core::Device;
use r2l_api::builders::sampler::EnvPoolType;
use r2l_api::builders::sampler_hooks2::{EvaluatorNormalizerOptions, EvaluatorOptions};
use r2l_api::test_utils::run_gym_episodes;
use r2l_core::agents::Agent;
use r2l_core::{Algorithm, on_policy_algorithm::LearningSchedule};
use r2l_gym::GymEnvBuilder;

const NUM_ENVIRONMENTS: usize = 10;
