// TODO: egui app here, should be different from the ratatui app in that the learning will not
// happen on a separate thread, just to illustrate how to use the training hooks

mod table;

use egui::{Pos2, Rect, UiBuilder};
use egui_plot::{Legend, Line, Plot, PlotPoint, PlotPoints};
use r2l_examples::{EventBox, PPOProgress, train_ppo};
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
    fn new(cc: &eframe::CreationContext<'_>, rx: Receiver<EventBox>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        // Self::default()
        Self {
            rx,
            recent_table: UpdateTable::default(),
            best_table: UpdateTable::default(),
            rollout_rewards_avg: vec![],
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_secs_f64(0.25));
        loop {
            let maybe_event = self.rx.try_recv();
            let Ok(event) = maybe_event else {
                break;
            };
            let Ok(progress) = event.downcast::<PPOProgress>() else {
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

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 240.0]) // wide enough for the drag-drop overlay text
            .with_drag_and_drop(true),
        ..Default::default()
    };
    let (event_tx, event_rx): (Sender<EventBox>, Receiver<EventBox>) = mpsc::channel();
    std::thread::spawn(move || match train_ppo(event_tx) {
        Ok(()) => {
            println!("ppo trainted normally")
        }
        Err(err) => {
            eprintln!("ppo was not trained normally, err: {err}")
        }
    });
    eframe::run_native(
        "R2L tui example",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc, event_rx)))),
    )
}
