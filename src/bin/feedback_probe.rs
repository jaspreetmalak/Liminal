//! Focused feedback ablation; never reads or writes the language checkpoint.

fn main() {
    if let Err(error) = liminal::probe::cli(std::env::args().skip(1).collect()) {
        eprintln!("feedback probe: {error}");
        std::process::exit(1);
    }
}
