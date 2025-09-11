use crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use r2l_examples::{EventBox, PPOProgress, train_ppo};
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

fn mean(numbers: &[f32]) -> f32 {
    let sum: f32 = numbers.iter().sum();
    sum / numbers.len() as f32
}

fn ppo_progres_to_table<'a>(ppo_progress: &'a PPOProgress, name: &'a str) -> Table<'a> {
    let block = Block::bordered().title(name).border_set(border::THICK);
    let widths = [Constraint::Percentage(50), Constraint::Percentage(50)];
    let entropy_loss = mean(&ppo_progress.entropy_losses);
    let value_loss = mean(&ppo_progress.value_losses);
    let policy_loss = mean(&ppo_progress.policy_losses);
    let clip_fraction = mean(&ppo_progress.clip_fractions);
    let rows = vec![
        // Row::new(vec!["approx_kl".into(), ppo_progress.approx_kl.to_string()]),
        Row::new(vec![
            "Avarage reward".into(),
            ppo_progress.avarage_reward.to_string(),
        ]),
        Row::new(vec!["Clip fraction".into(), clip_fraction.to_string()]),
        Row::new(vec![
            "Clip range".into(),
            ppo_progress.clip_range.to_string(),
        ]),
        Row::new(vec!["Policy gradient loss".into(), policy_loss.to_string()]),
        Row::new(vec!["Entropy loss".into(), entropy_loss.to_string()]),
        Row::new(vec!["Value loss".into(), value_loss.to_string()]),
        // Row::new(vec![
        //     "explained_variance".into(),
        //     ppo_progress.explained_variance.to_string(),
        // ]),
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
    pub fn run(mut self, terminal: &mut DefaultTerminal, rx: Receiver<EventBox>) -> io::Result<()> {
        while !self.exit {
            let event = rx.recv().unwrap();
            event
                .downcast::<PPOProgress>()
                .map(|progress| {
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
            ppo_progres_to_table(latest_update, "Latest update").render(latest_stat_area, buf);
            ppo_progres_to_table(best_update, "Best update").render(best_stat_area, buf);
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

fn handle_input_events(tx: mpsc::Sender<EventBox>) {
    if let crossterm::event::Event::Key(key_event) = crossterm::event::read().unwrap() {
        tx.send(Box::new(key_event)).unwrap()
    }
}

fn main() -> io::Result<()> {
    let (event_tx, event_rx): (Sender<EventBox>, Receiver<EventBox>) = mpsc::channel();
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
