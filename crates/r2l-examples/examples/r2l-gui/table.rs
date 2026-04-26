use egui::{Id, Margin};
use egui_table::{Column, HeaderCellInfo};
use r2l_api::PPOStats;

#[derive(Default)]
pub struct UpdateTable {
    pub progress: PPOStats,
    pub clip_range: f32,
}

impl UpdateTable {
    pub fn set_progress(&mut self, progress: PPOStats) {
        self.progress = progress
    }

    fn label_by_idx(&self, row_idx: u64, col_idx: usize) -> String {
        macro_rules! select_row_or_col {
            ($row:expr, $col:expr) => {
                if col_idx == 0 { $row } else { $col }
            };
        }

        let clip_fractions = self
            .progress
            .batch_stats
            .iter()
            .map(|s| s.clip_fraction)
            .collect::<Vec<_>>();

        let entropy_losses = self
            .progress
            .batch_stats
            .iter()
            .map(|s| s.entropy_loss)
            .collect::<Vec<_>>();

        let policy_losses = self
            .progress
            .batch_stats
            .iter()
            .map(|s| s.policy_loss)
            .collect::<Vec<_>>();

        let value_losses = self
            .progress
            .batch_stats
            .iter()
            .map(|s| s.value_loss)
            .collect::<Vec<_>>();

        let approx_kl = self
            .progress
            .batch_stats
            .iter()
            .map(|s| s.approx_kl)
            .collect::<Vec<_>>();

        match row_idx {
            0 => select_row_or_col!("clip_fractions".to_owned(), format!("{:?}", clip_fractions)),
            1 => select_row_or_col!("entropy_losses".to_owned(), format!("{:?}", entropy_losses)),
            2 => select_row_or_col!("policy_losses".to_owned(), format!("{:?}", policy_losses)),
            3 => select_row_or_col!("value_losses".to_owned(), format!("{:?}", value_losses)),
            4 => select_row_or_col!("clip_range".to_owned(), format!("{:?}", self.clip_range)),
            5 => select_row_or_col!("approx_kl".to_owned(), format!("{:?}", approx_kl)),
            6 => select_row_or_col!("explained_variance".to_owned(), format!("To be added")),
            7 => select_row_or_col!("progress".to_owned(), format!("To be added")),
            8 => select_row_or_col!(
                "std".to_owned(),
                self.progress
                    .std
                    .map(|std| std.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            ),
            9 => select_row_or_col!(
                "average_reward".to_owned(),
                format!("{:?}", self.progress.average_reward)
            ),
            10 => select_row_or_col!(
                "learning_rate".to_owned(),
                format!("{:?}", self.progress.learning_rate)
            ),
            _ => todo!(),
        }
    }
}

impl egui_table::TableDelegate for UpdateTable {
    fn prepare(&mut self, _info: &egui_table::PrefetchInfo) {}

    fn header_cell_ui(&mut self, ui: &mut egui::Ui, cell: &egui_table::HeaderCellInfo) {
        egui::Grid::new("settings").show(ui, |ui| {
            let HeaderCellInfo { row_nr, .. } = *cell;
            if row_nr == 0 {
                ui.label("Data name");
            } else {
                ui.label("Data");
            }
            ui.end_row();
        });
    }

    fn cell_ui(&mut self, ui: &mut egui::Ui, cell: &egui_table::CellInfo) {
        let egui_table::CellInfo { row_nr, col_nr, .. } = *cell;
        if row_nr % 2 == 1 {
            ui.painter()
                .rect_filled(ui.max_rect(), 0.0, ui.visuals().faint_bg_color);
        }
        let label = self.label_by_idx(row_nr, col_nr);
        egui::Frame::NONE
            .inner_margin(Margin::symmetric(4, 0))
            .show(ui, |ui| {
                ui.label(label);
            });
    }
}

impl UpdateTable {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let cols = vec![
            Column::new(500.).resizable(false),
            Column::new(500.).resizable(false),
        ];
        let id_salt = Id::new("table_demo");
        let table = egui_table::Table::new()
            .id_salt(id_salt)
            .num_rows(11)
            .columns(cols)
            .auto_size_mode(egui_table::AutoSizeMode::Always);
        table.show(ui, self);
    }
}
