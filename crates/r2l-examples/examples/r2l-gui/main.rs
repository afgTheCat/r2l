// TODO: egui app here, should be different from the ratatui app in that the learning will not
// happen on a separate thread, just to illustrate how to use the training hooks

mod table;

use burn::{
    backend::{Autodiff, NdArray},
    optim::AdamWConfig,
};
use egui::{Pos2, Rect, UiBuilder};
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};
use r2l_api::hooks::ppo::{PPOHookBuilder, PPOStats};
use r2l_burn_lm::{
    distributions::diagonal_distribution::DiagGaussianDistribution,
    learning_module::{ParalellActorCriticLM, ParalellActorModel},
};
use r2l_core::env_builder::EnvBuilder;
use r2l_core::on_policy_algorithm::{
    DefaultOnPolicyAlgorightmsHooks5, LearningSchedule, OnPolicyAlgorithm,
};
use r2l_core::sampler::{FinalSampler, Location, buffer::StepTrajectoryBound};
use r2l_examples::EventBox;
use std::sync::Arc;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use crate::table::UpdateTable;

type BurnBackend = Autodiff<NdArray>;

struct App {
    recent_table: UpdateTable,
    best_table: UpdateTable,
    rx: Receiver<EventBox>,
    rollout_rewards_avg: Vec<f32>,
    total_rollouts: usize,
    current_rollout: usize,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        rx: Receiver<EventBox>,
        total_rollouts: usize,
        clip_range: f32,
    ) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        // Self::default()
        Self {
            rx,
            recent_table: UpdateTable {
                clip_range,
                ..Default::default()
            },
            best_table: UpdateTable {
                clip_range,
                progress: Default::default(),
            },
            rollout_rewards_avg: vec![],
            total_rollouts,
            current_rollout: 0,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_secs_f64(0.25));
        loop {
            let event = self.rx.try_recv();
            let Ok(event) = event else {
                break;
            };
            let Ok(progress) = event.downcast::<PPOStats>() else {
                break;
            };
            let avg_rewards = progress.avarage_reward;
            self.rollout_rewards_avg.push(avg_rewards);
            self.recent_table.set_progress(*progress.clone());
            if self
                .rollout_rewards_avg
                .iter()
                .all(|rew| *rew <= avg_rewards)
            {
                self.best_table.set_progress(*progress);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let availible_rect = Rect::from_min_max(
                Pos2::new(0., 0.),
                Pos2::new(ui.available_width(), ui.available_height()),
            );
            let (tables, other_widgets) = availible_rect.split_top_bottom_at_fraction(0.5);
            let (mut recent_table_rect, mut best_table_rect) =
                tables.split_left_right_at_fraction(0.5);

            // TODO: maybe we should shrink it within the rect
            recent_table_rect = recent_table_rect.shrink(10.);
            ui.scope_builder(UiBuilder::new().max_rect(recent_table_rect), |ui| {
                self.recent_table.ui(ui)
            });

            best_table_rect = best_table_rect.shrink(10.);
            ui.scope_builder(UiBuilder::new().max_rect(best_table_rect), |ui| {
                self.best_table.ui(ui)
            });

            let (progress_bar, plot) = other_widgets.split_top_bottom_at_fraction(0.1);
            ui.scope_builder(UiBuilder::new().max_rect(plot), |ui| {
                Plot::new("Plot")
                    .legend(Legend::default())
                    .show(ui, |plot_ui| {
                        let plot_points = self
                            .rollout_rewards_avg
                            .iter()
                            .enumerate()
                            .map(|(idx, avg_rew)| PlotPoint::from([idx as f64, *avg_rew as f64]))
                            .collect();
                        plot_ui
                            .line(Line::new("curve", PlotPoints::Owned(plot_points)).name("curve"));
                    })
            });
        });
    }
}

const ENT_COEFF: f32 = 0.001;
const MAX_GRAD_NORM: f32 = 0.5;
const TARGET_KL: f32 = 0.01;
const ENV_NAME: &str = "Pendulum-v1";

pub fn train_ppo2(
    tx: Sender<PPOStats>,
    total_rollouts: usize,
    clip_range: f32,
) -> anyhow::Result<()> {
    let ppo_hook = PPOHookBuilder::new()
        .with_entropy_coeff(ENT_COEFF)
        .with_gradient_clipping(Some(MAX_GRAD_NORM))
        .with_target_kl(Some(TARGET_KL))
        .build(tx);
    let env_builder = EnvBuilder::Homogenous {
        builder: Arc::new(r2l_gym::GymEnvBuilder::new(ENV_NAME)),
        n_envs: 5,
    };
    let sampler = FinalSampler::build(
        env_builder,
        StepTrajectoryBound::new(2048),
        None,
        Location::Vec,
    );
    let env_description = sampler.env_description();
    let action_size = env_description.action_space.size();
    let observation_size = env_description.observation_space.size();
    let policy_layers = &[observation_size, 64, 64, action_size];
    let value_layers = &[observation_size, 64, 64, 1];
    let distr: DiagGaussianDistribution<BurnBackend> =
        DiagGaussianDistribution::build(policy_layers);
    let value_net = r2l_burn_lm::sequential::Sequential::build(value_layers);
    let model = ParalellActorModel::new(distr, value_net);
    let lm = ParalellActorCriticLM::new(model, AdamWConfig::new().init());
    let agent = r2l_agents::burn_agents::ppo::BurnPPO::new(
        r2l_agents::burn_agents::ppo::BurnPPOCore::new(lm, clip_range, 64, 0.98, 0.8),
        ppo_hook,
    );
    let mut algo = OnPolicyAlgorithm {
        sampler,
        agent,
        hooks: DefaultOnPolicyAlgorightmsHooks5::new(LearningSchedule::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        }),
    };
    algo.train()
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 240.0]) // wide enough for the drag-drop overlay text
            .with_drag_and_drop(true),
        ..Default::default()
    };
    let (event_tx, event_rx): (Sender<EventBox>, Receiver<EventBox>) = mpsc::channel();
    let (update_tx, update_rx): (Sender<PPOStats>, Receiver<PPOStats>) = mpsc::channel();
    let tx_to_events = event_tx.clone();
    std::thread::spawn(move || {
        while let Ok(update) = update_rx.recv() {
            tx_to_events.send(Box::new(update)).unwrap();
        }
    });
    let total_rollouts = 300;
    let clip_range = 0.2;
    std::thread::spawn(
        move || match train_ppo2(update_tx, total_rollouts, clip_range) {
            Ok(()) => {
                println!("ppo trainted normally")
            }
            Err(err) => {
                eprintln!("ppo was not trained normally, err: {err}")
            }
        },
    );
    eframe::run_native(
        "R2L tui example",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc, event_rx, total_rollouts, clip_range)))),
    )
}
