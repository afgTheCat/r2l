pub fn mean(numbers: &[f32]) -> f32 {
    let sum: f32 = numbers.iter().sum();
    sum / numbers.len() as f32
}

pub fn fmt_stat(x: f32) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x.abs() < 0.001 {
        format!("{x:.2e}")
    } else {
        format!("{x:.4}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}
