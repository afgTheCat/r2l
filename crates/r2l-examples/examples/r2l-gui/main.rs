// TODO: egui app here, should be different from the ratatui app in that the learning will not
// happen on a separate thread, just to illustrate how to use the training hooks

mod table;

use egui::{Pos2, Rect, UiBuilder};
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};
use r2l_api::builders::ppo::algorithm::PPOAlgorithmBuilder;
use r2l_api::hooks::on_policy::LearningSchedule;
use r2l_api::hooks::ppo::PPOStats;
use r2l_core::sampler::{Location, buffer::StepTrajectoryBound};
use r2l_examples::EventBox;
use r2l_gym::GymEnvBuilder;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use crate::table::UpdateTable;

struct App {
    recent_table: UpdateTable,
    best_table: UpdateTable,
    rx: Receiver<EventBox>,
    rollout_rewards_avg: Vec<f32>,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>, rx: Receiver<EventBox>, clip_range: f32) -> Self {
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
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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

pub fn train_ppo(
    tx: Sender<PPOStats>,
    total_rollouts: usize,
    clip_range: f32,
) -> anyhow::Result<()> {
    // TODO: The generic here is ugly
    let ppo_builder = PPOAlgorithmBuilder::<GymEnvBuilder>::new(ENV_NAME, 10)
        .with_burn()
        .with_entropy_coeff(ENT_COEFF)
        .with_gradient_clipping(Some(MAX_GRAD_NORM))
        .with_target_kl(Some(TARGET_KL))
        .with_bound(StepTrajectoryBound::new(2048))
        .with_location(Location::Vec)
        .with_clip_range(clip_range)
        .with_learning_schedule(LearningSchedule::RolloutBound {
            total_rollouts,
            current_rollout: 0,
        })
        .with_reporter(Some(tx));
    let mut ppo = ppo_builder.build()?;
    ppo.train()
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
        move || match train_ppo(update_tx, total_rollouts, clip_range) {
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
        Box::new(|cc| Ok(Box::new(App::new(cc, event_rx, clip_range)))),
    )
}
