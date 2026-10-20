//! Reproducible SplitMix64 randomness; not suitable for cryptography.

pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
    }

    pub(crate) fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    pub(crate) fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "random index requires a nonempty range");
        let bound = upper as u64;
        let rejection_threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= rejection_threshold {
                return (value % bound) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn deterministic_stream_and_valid_ranges() {
        let mut first = Rng::new(42);
        let mut second = Rng::new(42);
        for _ in 0..1000 {
            let value = first.next_f64();
            assert_eq!(value, second.next_f64());
            assert!((0.0..1.0).contains(&value));
            assert!(first.index(7) < 7);
            second.index(7);
        }
    }
}
