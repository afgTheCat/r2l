use egui::{Id, Margin};
use egui_table::{Column, HeaderCellInfo};
use r2l_examples::PPOProgress;

#[derive(Default)]
pub struct UpdateTable {
    progress: PPOProgress,
}

impl UpdateTable {
    pub fn set_progress(&mut self, progress: PPOProgress) {
        self.progress = progress
    }

    fn label_by_idx(&self, row_idx: u64, col_idx: usize) -> String {
        macro_rules! select_row_or_col {
            ($row:expr, $col:expr) => {
                if col_idx == 0 { $row } else { $col }
            };
        }

        match row_idx {
            0 => select_row_or_col!(
                "clip_fractions".to_owned(),
                format!("{:?}", self.progress.clip_fractions)
            ),
            1 => select_row_or_col!(
                "entropy_losses".to_owned(),
                format!("{:?}", self.progress.entropy_losses)
            ),
            2 => select_row_or_col!(
                "policy_losses".to_owned(),
                format!("{:?}", self.progress.policy_losses)
            ),
            3 => select_row_or_col!(
                "value_losses".to_owned(),
                format!("{:?}", self.progress.value_losses)
            ),
            4 => select_row_or_col!(
                "clip_range".to_owned(),
                format!("{:?}", self.progress.clip_range)
            ),
            5 => select_row_or_col!(
                "approx_kl".to_owned(),
                format!("{:?}", self.progress.approx_kl)
            ),
            6 => select_row_or_col!(
                "explained_variance".to_owned(),
                format!("{:?}", self.progress.explained_variance)
            ),
            7 => select_row_or_col!(
                "progress".to_owned(),
                format!("{:?}", self.progress.progress)
            ),
            8 => select_row_or_col!("std".to_owned(), format!("{:?}", self.progress.std)),
            9 => select_row_or_col!(
                "avarage_reward".to_owned(),
                format!("{:?}", self.progress.avarage_reward)
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
