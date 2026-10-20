//! Frozen audit of the current XL language checkpoint; no training.
fn main() {
    if let Err(error) = liminal::generation_probe::cli(std::env::args().skip(1).collect()) {
        eprintln!("generation probe: {error}");
        std::process::exit(1);
    }
}
