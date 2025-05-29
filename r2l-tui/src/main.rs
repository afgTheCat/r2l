// TODO: probably needs to make this event driven
mod ppo;

use candle_core::{DType, Tensor};
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ppo::train_ppo;
use ratatui::{
    DefaultTerminal, Frame,
    buffer::Buffer,
    layout::{Constraint, Layout, Rect},
    style::{Color, Style, Stylize},
    symbols::{self, border},
    text::Line,
    widgets::{Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Row, Table, Widget},
};
use std::sync::mpsc::Receiver;
use std::{f64, io, sync::mpsc};

const ENV_NAME: &str = "Pendulum-v1";

fn mean(numbers: &[f32]) -> f32 {
    let sum: f32 = numbers.iter().sum();
    sum / numbers.len() as f32
}

#[derive(Debug, Default)]
struct App {
    exit: bool,
    latest_update: Option<PPOProgress>,
    best_update: Option<PPOProgress>,
    rollout_rewards_avg: Vec<f32>,
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
        self.draw_progress_bar(progress_bar_area, buf);
        self.draw_chart(chart_area, buf);
    }
}

impl App {
    pub fn run(mut self, terminal: &mut DefaultTerminal, rx: Receiver<PPOEvent>) -> io::Result<()> {
        while !self.exit {
            match rx.recv().unwrap() {
                PPOEvent::Input(key_event) => self.handle_events(key_event)?,
                PPOEvent::Progress(progress) => self.handle_progress(progress),
            }
            terminal.draw(|frame| self.draw(frame))?;
        }
        Ok(())
    }

    fn handle_progress(&mut self, progress: PPOProgress) {
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

    fn draw_statistics(&self, statistics_area: Rect, buf: &mut Buffer) {
        if let (Some(latest_update), Some(best_update)) = (&self.latest_update, &self.best_update) {
            let vertical_area =
                Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]);
            let [latest_stat_area, best_stat_area] = vertical_area.areas(statistics_area);
            latest_update
                .to_table("Latest update")
                .render(latest_stat_area, buf);
            best_update
                .to_table("Best update")
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

    fn draw_progress_bar(&self, pb_area: Rect, buf: &mut Buffer) {
        let Some(progress) = self.latest_update.as_ref().map(|x| x.progress) else {
            return;
        };
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

#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct PPOProgress {
    clip_fractions: Vec<f32>,
    entropy_losses: Vec<f32>,
    policy_losses: Vec<f32>,
    value_losses: Vec<f32>,
    clip_range: f32,
    approx_kl: f32,
    explained_variance: f32,
    progress: f64,
    std: f32,
    avarage_reward: f32,
    learning_rate: f64,
}

impl PPOProgress {
    pub fn clear(&mut self) -> Self {
        std::mem::take(self)
    }

    pub fn collect_batch_data(
        &mut self,
        // clip_range: f32,
        ratio: &Tensor,
        entropy_loss: &Tensor,
        value_loss: &Tensor,
        policy_loss: &Tensor,
    ) -> candle_core::Result<()> {
        let clip_fraction = (ratio - 1.)?
            .abs()?
            .gt(self.clip_range)?
            .to_dtype(DType::F32)?
            .mean_all()?
            .to_scalar::<f32>()?;
        self.clip_fractions.push(clip_fraction);
        self.entropy_losses.push(entropy_loss.to_scalar()?);
        self.value_losses.push(value_loss.to_scalar()?);
        self.policy_losses.push(policy_loss.to_scalar()?);
        Ok(())
    }
}

impl PPOProgress {
    fn to_table<'a>(&'a self, name: &'a str) -> Table<'a> {
        let block = Block::bordered().title(name).border_set(border::THICK);
        let widths = [Constraint::Percentage(50), Constraint::Percentage(50)];
        let entropy_loss = mean(&self.entropy_losses);
        let value_loss = mean(&self.value_losses);
        let policy_loss = mean(&self.policy_losses);
        let clip_fraction = mean(&self.clip_fractions);
        let rows = vec![
            // Row::new(vec!["approx_kl".into(), self.approx_kl.to_string()]),
            Row::new(vec![
                "Avarage reward".into(),
                self.avarage_reward.to_string(),
            ]),
            Row::new(vec!["Clip fraction".into(), clip_fraction.to_string()]),
            Row::new(vec!["Clip range".into(), self.clip_range.to_string()]),
            Row::new(vec!["Policy gradient loss".into(), policy_loss.to_string()]),
            Row::new(vec!["Entropy loss".into(), entropy_loss.to_string()]),
            Row::new(vec!["Value loss".into(), value_loss.to_string()]),
            // Row::new(vec![
            //     "explained_variance".into(),
            //     self.explained_variance.to_string(),
            // ]),
            Row::new(vec!["Learning rate".into(), self.learning_rate.to_string()]),
            Row::new(vec!["Standard deviation".into(), self.std.to_string()]),
        ];
        Table::new(rows, widths).block(block)
    }
}

pub enum PPOEvent {
    Progress(PPOProgress),
    Input(KeyEvent),
}

fn handle_input_events(tx: mpsc::Sender<PPOEvent>) {
    if let crossterm::event::Event::Key(key_event) = crossterm::event::read().unwrap() {
        tx.send(PPOEvent::Input(key_event)).unwrap()
    }
}

fn main() -> io::Result<()> {
    let (event_tx, event_rx) = mpsc::channel::<PPOEvent>();
    let tx_to_input_events = event_tx.clone();
    std::thread::spawn(move || {
        handle_input_events(tx_to_input_events);
    });

    std::thread::spawn(move || match train_ppo(event_tx) {
        Ok(()) => {
            println!("ppo trainted normally")
        }
        Err(err) => {
            eprintln!("ppo was not trained normally, err: {err}")
        }
    });
    let mut terminal = ratatui::init();
    let app_result = App::default().run(&mut terminal, event_rx);
    // learning_t.join().unwrap();
    ratatui::restore();
    app_result
}
