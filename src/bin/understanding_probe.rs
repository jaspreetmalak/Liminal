//! Frozen fact-consistency and natural-context diagnostics; no training.
fn main() {
    if let Err(error) = liminal::understanding_probe::cli(std::env::args().skip(1).collect()) {
        eprintln!("understanding probe: {error}");
        std::process::exit(1);
    }
}
