//! Frozen text/context diagnostic; no optimizer updates.
fn main() {
    if let Err(error) = liminal::text_probe::cli(std::env::args().skip(1).collect()) {
        eprintln!("text probe: {error}");
        std::process::exit(1);
    }
}
