use std::{
    path::Path,
    time::{Duration, Instant},
};

use r2l_core::error::Error;
use serde::{Deserialize, Serialize, Serializer};

use crate::{
    constants::TRAINING_TIMINGS_FILE, csv_artifact::CsvArtifact,
    hooks::progress::SharedTrainingProgress,
};

#[derive(Debug, Clone, Copy)]
pub(super) enum Phase {
    Rollout,
    Training,
    Evaluation,
}

struct RecordedPhase {
    phase: Phase,
    started: Instant,
}

impl RecordedPhase {
    fn new(phase: Phase) -> Self {
        Self {
            phase,
            started: Instant::now(),
        }
    }
}

#[derive(Default)]
struct TrainingLoopTimings {
    collection: Duration,
    training: Duration,
    evaluation: Duration,
}

#[derive(Serialize, Deserialize)]
struct TimingRecord {
    rollout: usize,
    #[serde(serialize_with = "TimingRecord::serialize_milliseconds")]
    collect_ms: f64,
    #[serde(serialize_with = "TimingRecord::serialize_milliseconds")]
    learn_ms: f64,
    #[serde(serialize_with = "TimingRecord::serialize_milliseconds")]
    evaluate_ms: f64,
    #[serde(serialize_with = "TimingRecord::serialize_milliseconds")]
    total_ms: f64,
}

impl TimingRecord {
    fn serialize_milliseconds<S: Serializer>(
        value: &f64,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.collect_str(&format_args!("{value:.3}"))
    }
}

pub(crate) struct EnabledTimingRecorder {
    file: CsvArtifact<TimingRecord>,
    progress: SharedTrainingProgress,
    timings: TrainingLoopTimings,
    phase_recorder: Option<RecordedPhase>,
    training_started: Instant,
}

impl EnabledTimingRecorder {
    fn finish_current_phase_recording(&mut self) {
        let Some(RecordedPhase { phase, started }) = self.phase_recorder.take() else {
            return;
        };
        let elapsed = started.elapsed();
        match phase {
            Phase::Rollout => self.timings.collection = elapsed,
            Phase::Training => self.timings.training = elapsed,
            Phase::Evaluation => self.timings.evaluation = elapsed,
        }
    }

    fn record_new_phase(&mut self, phase: Phase) {
        self.phase_recorder = Some(RecordedPhase::new(phase));
    }

    fn flush(&mut self) -> Result<(), Error> {
        let completed_rollouts = self.progress.borrow().completed_rollouts();
        let timings = std::mem::take(&mut self.timings);
        self.file.write(&TimingRecord {
            rollout: completed_rollouts,
            collect_ms: timings.collection.as_secs_f64() * 1000.0,
            learn_ms: timings.training.as_secs_f64() * 1000.0,
            evaluate_ms: timings.evaluation.as_secs_f64() * 1000.0,
            total_ms: self.training_started.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

pub(crate) enum TimingRecorder {
    Disabled,
    Enabled(EnabledTimingRecorder),
}

impl TimingRecorder {
    pub(crate) fn disabled() -> Self {
        Self::Disabled
    }

    /// Creates a CSV recorder for training phase durations.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory in which the timings CSV is created.
    /// * `progress` - Training progress used to label each timings row.
    pub(crate) fn create(
        output_dir: &Path,
        progress: SharedTrainingProgress,
    ) -> Result<Self, Error> {
        std::fs::create_dir_all(output_dir)?;
        let file = CsvArtifact::open(&output_dir.join(TRAINING_TIMINGS_FILE))?;
        Ok(Self::Enabled(EnabledTimingRecorder {
            file,
            progress,
            timings: TrainingLoopTimings::default(),
            phase_recorder: None,
            training_started: Instant::now(),
        }))
    }

    pub(super) fn init(&mut self) {
        if let Self::Enabled(recorder) = self {
            recorder.training_started = Instant::now();
            recorder.timings = TrainingLoopTimings::default();
            recorder.phase_recorder = None;
        }
        self.record_new_phase(Phase::Rollout);
    }

    pub(super) fn record_new_phase(&mut self, phase: Phase) {
        match self {
            Self::Disabled => {}
            Self::Enabled(enabled) => enabled.record_new_phase(phase),
        }
    }

    pub(super) fn finish_current_phase_recording(&mut self) {
        match self {
            Self::Disabled => {}
            Self::Enabled(enabled) => enabled.finish_current_phase_recording(),
        }
    }

    pub(super) fn flush(&mut self) -> Result<(), Error> {
        match self {
            Self::Disabled => Ok(()),
            Self::Enabled(enabled) => enabled.flush(),
        }
    }
}
