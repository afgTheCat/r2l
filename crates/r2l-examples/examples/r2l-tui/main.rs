use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use r2l_api::algorithm::ppo::candle::PPOCandleAlgorithmBuiler;
use r2l_api::hooks::ppo::PPOStats;
use r2l_core::on_policy_algorithm::LearningSchedule;
use r2l_core::sampler::Location;
use r2l_core::sampler::buffer::StepTrajectoryBound;
use r2l_examples::EventBox;
use r2l_gym::GymEnvBuilder;
use ratatui::{
    DefaultTerminal, Frame,
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    symbols::{self, border},
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Row, Table, Widget},
};
use std::sync::mpsc::{Receiver, Sender};
use std::{f64, io, sync::mpsc};

const ENT_COEFF: f32 = 0.001;
const MAX_GRAD_NORM: f32 = 0.5;
const TARGET_KL: f32 = 0.01;
const ENV_NAME: &str = "Pendulum-v1";

fn mean(numbers: &[f32]) -> f32 {
    let sum: f32 = numbers.iter().sum();
    sum / numbers.len() as f32
}

#[derive(Debug)]
struct App {
    exit: bool,
    latest_update: Option<PPOStats>,
    rx: Receiver<EventBox>,
    best_update: Option<PPOStats>,
    rollout_rewards_avg: Vec<f32>,
    clip_range: f32,
    total_rollouts: usize,
    current_rollout: usize,
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer)
    where
        Self: Sized,
    {
        let horizontal_area = Layout::vertical([
            Constraint::Percentage(40),
            Constraint::Percentage(10),
            Constraint::Percentage(50),
        ]);
        let [statistics_area, progress_bar_area, chart_area] = horizontal_area.areas(area);
        self.draw_statistics(statistics_area, buf);
        let progress = self.current_rollout as f64 / self.total_rollouts as f64;
        self.draw_progress_bar(progress_bar_area, buf, progress);
        self.draw_chart(chart_area, buf);
    }
}

impl App {
    fn new(total_rollouts: usize, clip_range: f32, rx: Receiver<EventBox>) -> Self {
        Self {
            total_rollouts,
            clip_range,
            rx,
            best_update: None,
            latest_update: None,
            rollout_rewards_avg: vec![],
            current_rollout: 0,
            exit: false,
        }
    }

    pub fn run(mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            let event = self.rx.recv().unwrap();
            event
                .downcast::<PPOStats>()
                .map(|progress| {
                    self.current_rollout += 1;
                    self.handle_progress(*progress);
                })
                .or_else(|event| {
                    event.downcast::<KeyEvent>().map(|key_event| {
                        self.handle_events(*key_event).unwrap();
                    })
                })
                .unwrap_or_else(|_| unreachable!("Unknown event type received"));
            terminal.draw(|frame| self.draw(frame))?;
        }
        Ok(())
    }

    fn handle_progress(&mut self, progress: PPOStats) {
        self.rollout_rewards_avg.push(progress.avarage_reward);
        match &self.best_update {
            None => self.best_update = Some(progress.clone()),
            Some(current_best) if current_best.avarage_reward < progress.avarage_reward => {
                self.best_update = Some(progress.clone())
            }
            _ => {}
        }
        self.latest_update = Some(progress)
    }

    fn draw(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.area());
    }

    fn handle_events(&mut self, key_event: crossterm::event::KeyEvent) -> io::Result<()> {
        if key_event.kind == KeyEventKind::Press {
            self.handle_key_event(key_event)
        }
        Ok(())
    }

    // maybe something more eventually
    fn handle_key_event(&mut self, key_event: KeyEvent) {
        if let KeyCode::Char('q') = key_event.code {
            self.exit()
        }
    }

    fn exit(&mut self) {
        self.exit = true;
    }

    fn ppo_progress_to_table<'a>(&self, ppo_progress: &'a PPOStats, name: &'a str) -> Table<'a> {
        let block = Block::bordered().title(name).border_set(border::THICK);
        let widths = [Constraint::Percentage(50), Constraint::Percentage(50)];
        let entropy_loss = mean(
            &ppo_progress
                .batch_stats
                .iter()
                .map(|s| s.entropy_loss)
                .collect::<Vec<_>>(),
        );
        let value_loss = mean(
            &ppo_progress
                .batch_stats
                .iter()
                .map(|s| s.value_loss)
                .collect::<Vec<_>>(),
        );
        let policy_loss = mean(
            &ppo_progress
                .batch_stats
                .iter()
                .map(|s| s.policy_loss)
                .collect::<Vec<_>>(),
        );
        let clip_fraction = mean(
            &ppo_progress
                .batch_stats
                .iter()
                .map(|s| s.clip_fraction)
                .collect::<Vec<_>>(),
        );
        let rows = vec![
            // Row::new(vec!["approx_kl".into(), ppo_progress.approx_kl.to_string()]),
            Row::new(vec![
                "Avarage reward".into(),
                ppo_progress.avarage_reward.to_string(),
            ]),
            Row::new(vec!["Clip fraction".into(), clip_fraction.to_string()]),
            Row::new(vec!["Clip range".into(), self.clip_range.to_string()]),
            Row::new(vec!["Policy gradient loss".into(), policy_loss.to_string()]),
            Row::new(vec!["Entropy loss".into(), entropy_loss.to_string()]),
            Row::new(vec!["Value loss".into(), value_loss.to_string()]),
            Row::new(vec!["explained_variance".into(), "to be added".to_string()]),
            Row::new(vec![
                "Learning rate".into(),
                ppo_progress.learning_rate.to_string(),
            ]),
            Row::new(vec![
                "Standard deviation".into(),
                ppo_progress.std.to_string(),
            ]),
        ];
        Table::new(rows, widths).block(block)
    }

    fn draw_statistics(&self, statistics_area: Rect, buf: &mut Buffer) {
        if let (Some(latest_update), Some(best_update)) = (&self.latest_update, &self.best_update) {
            let vertical_area =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]);
            let [latest_stat_area, best_stat_area] = vertical_area.areas(statistics_area);
            self.ppo_progress_to_table(latest_update, "Latest update")
                .render(latest_stat_area, buf);
            self.ppo_progress_to_table(best_update, "Best update")
                .render(best_stat_area, buf);
        }
    }

    fn draw_chart(&self, chart_area: Rect, buf: &mut Buffer) {
        let rewards: Vec<_> = self
            .rollout_rewards_avg
            .iter()
            .enumerate()
            .map(|(i, d)| (i as f64, *d as f64))
            .collect();
        if rewards.is_empty() {
            Line::from("Waiting for data to render").render(chart_area, buf);
        } else {
            let x_bounds = [0., rewards.last().unwrap().0];
            let y_bounds = [
                rewards
                    .iter()
                    .fold(f64::INFINITY, |acc, (_, elem)| f64::min(acc, *elem)),
                rewards
                    .iter()
                    .fold(-f64::INFINITY, |acc, (_, elem)| f64::max(acc, *elem)),
            ];
            let labels = [
                format!("{:.2}", y_bounds[0]).bold(),
                format!("{:.2}", y_bounds[0] + (y_bounds[1] - y_bounds[0]) / 2.).bold(),
                format!("{:.2}", y_bounds[1]).bold(),
            ];
            let dataset = vec![
                Dataset::default()
                    .marker(symbols::Marker::Braille)
                    .style(Style::new().fg(Color::Blue))
                    .graph_type(GraphType::Line)
                    .data(&rewards),
            ];
            let chart = Chart::new(dataset)
                .block(
                    Block::default()
                        .title("Avarage rewards per rollout")
                        .borders(Borders::ALL),
                )
                .x_axis(
                    Axis::default()
                        .bounds(x_bounds)
                        .style(Style::default().gray()), // .bounds(x_bounds),
                )
                .y_axis(
                    Axis::default()
                        .style(Style::default().gray())
                        .bounds(y_bounds)
                        .labels(labels),
                )
                .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));
            chart.render(chart_area, buf);
        }
    }

    fn draw_progress_bar(&self, pb_area: Rect, buf: &mut Buffer, progress: f64) {
        let block = Block::bordered()
            .title(Line::from(" Learning Progress "))
            .border_set(border::THICK);
        let progress_bar = Gauge::default()
            .gauge_style(Style::default().fg(Color::Green))
            .block(block)
            .label(format!("Progress: {:.2}%", progress * 100_f64))
            .ratio(progress);
        progress_bar.render(pb_area, buf);
    }
}

fn handle_input_events(tx: mpsc::Sender<EventBox>) {
    std::thread::spawn(move || {
        loop {
            if let crossterm::event::Event::Key(key_event) = crossterm::event::read().unwrap() {
                tx.send(Box::new(key_event)).unwrap()
            }
        }
    });
}

pub fn train_ppo(
    tx: Sender<PPOStats>,
    total_rollouts: usize,
    clip_range: f32,
) -> anyhow::Result<()> {
    // TODO: The generic here is ugly
    let ppo_builder = PPOCandleAlgorithmBuiler::<GymEnvBuilder>::new(ENV_NAME, 10)
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

pub fn adapt_ppo_events(update_rx: Receiver<PPOStats>, tx_to_updates: Sender<EventBox>) {
    std::thread::spawn(move || {
        while let Ok(update) = update_rx.recv() {
            tx_to_updates.send(Box::new(update)).unwrap();
        }
    });
}

fn main() -> io::Result<()> {
    let (event_tx, event_rx) = mpsc::channel();
    let (update_tx, update_rx) = mpsc::channel();
    handle_input_events(event_tx.clone());
    adapt_ppo_events(update_rx, event_tx.clone());
    let total_rollouts = 300;
    let clip_range = 0.2;
    std::thread::spawn(
        move || match train_ppo(update_tx, total_rollouts, clip_range) {
            Ok(()) => {}
            Err(err) => {
                eprintln!("ppo was not trained normally, err: {err}")
            }
        },
    );
    let mut terminal = ratatui::init();
    let app_result = App::new(total_rollouts, clip_range, event_rx).run(&mut terminal);
    ratatui::restore();
    app_result
}
