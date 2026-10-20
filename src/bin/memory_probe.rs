//! Focused memory diagnosis using the single maintained Burn model.

fn main() {
    if let Err(error) = liminal::memory_probe::cli(std::env::args().skip(1).collect()) {
        eprintln!("memory probe: {error}");
        std::process::exit(1);
    }
}
