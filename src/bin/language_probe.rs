//! Bounded genuine FineWeb passage fitting with the single production model.

fn main() {
    if let Err(error) = liminal::language_probe::cli(std::env::args().skip(1).collect()) {
        eprintln!("FineWeb language probe: {error}");
        std::process::exit(1);
    }
}
