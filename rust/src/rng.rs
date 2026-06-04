use std::num::NonZeroU64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RngError {
    EmptyChoices,
    NoPositiveWeights,
    SampleSpaceOverflow,
}

trait RandomSource {
    fn next_u64(&mut self) -> u64;
}

struct Rng<S: RandomSource = SplitMix64> {
    source: S,
}

impl Rng<SplitMix64> {
    fn from_seed_u64(seed: u64) -> Self {
        Self {
            source: SplitMix64::from_seed_u64(seed),
        }
    }
}

impl<S: RandomSource> Rng<S> {
    fn from_source(source: S) -> Self {
        Self { source }
    }

    fn uniform_index(&mut self, len: usize) -> Result<usize, RngError> {
        let len = checked_choice_count(len)?;
        Ok(self.uniform_below_u64(len) as usize)
    }

    fn weighted_index(&mut self, weights: &[usize]) -> Result<usize, RngError> {
        let total = checked_weight_total(weights)?;
        let mut draw = self.uniform_below_u64(total) as usize;
        for (index, &weight) in weights.iter().enumerate() {
            if draw < weight {
                return Ok(index);
            }
            draw -= weight;
        }

        unreachable!("weighted draw must be less than the checked total weight")
    }

    fn uniform_below_u64(&mut self, upper: NonZeroU64) -> u64 {
        let draw_space = 1_u128 << 64;
        let upper = u128::from(upper.get());
        let accepted = (draw_space / upper) * upper;

        loop {
            let draw = u128::from(self.source.next_u64());
            if draw < accepted {
                return (draw % upper) as u64;
            }
        }
    }
}

fn sample_space_u64(value: usize) -> Result<u64, RngError> {
    u64::try_from(value).map_err(|_| RngError::SampleSpaceOverflow)
}

fn checked_choice_count(len: usize) -> Result<NonZeroU64, RngError> {
    NonZeroU64::new(sample_space_u64(len)?).ok_or(RngError::EmptyChoices)
}

fn checked_weight_total(weights: &[usize]) -> Result<NonZeroU64, RngError> {
    let mut total = 0_usize;
    for &weight in weights {
        total = total
            .checked_add(weight)
            .ok_or(RngError::SampleSpaceOverflow)?;
    }
    NonZeroU64::new(sample_space_u64(total)?).ok_or(RngError::NoPositiveWeights)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn from_seed_u64(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl RandomSource for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::{RandomSource, Rng, RngError, SplitMix64};

    struct FixedSource<const N: usize> {
        draws: [u64; N],
        index: usize,
    }

    impl<const N: usize> FixedSource<N> {
        fn new(draws: [u64; N]) -> Self {
            Self { draws, index: 0 }
        }
    }

    impl<const N: usize> RandomSource for FixedSource<N> {
        fn next_u64(&mut self) -> u64 {
            let draw = self.draws[self.index];
            self.index += 1;
            draw
        }
    }

    #[test]
    fn splitmix64_sequence_is_pinned() {
        let mut source = SplitMix64::from_seed_u64(0);
        assert_eq!(source.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(source.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(source.next_u64(), 0x06c4_5d18_8009_454f);

        let mut source = SplitMix64::from_seed_u64(0x0123_4567_89ab_cdef);
        assert_eq!(source.next_u64(), 0x157a_3807_a48f_aa9d);
        assert_eq!(source.next_u64(), 0xd573_529b_34a1_d093);
        assert_eq!(source.next_u64(), 0x2f90_b72e_996d_ccbe);
    }

    #[test]
    fn uniform_index_rejects_empty_choice_set() {
        let mut rng = Rng::from_seed_u64(0);
        assert_eq!(rng.uniform_index(0), Err(RngError::EmptyChoices));
    }

    #[test]
    fn uniform_index_rejects_outside_the_accepted_draw_space() {
        let mut rng = Rng::from_source(FixedSource::new([u64::MAX, 6]));
        assert_eq!(rng.uniform_index(3), Ok(0));
    }

    #[test]
    fn uniform_index_accepts_the_full_draw_space_when_divisible() {
        let mut rng = Rng::from_source(FixedSource::new([u64::MAX]));
        assert_eq!(rng.uniform_index(1), Ok(0));

        let mut rng = Rng::from_source(FixedSource::new([u64::MAX]));
        assert_eq!(rng.uniform_index(2), Ok(1));
    }

    #[test]
    fn weighted_index_rejects_no_positive_or_overflowing_weights() {
        let mut rng = Rng::from_seed_u64(0);
        assert_eq!(rng.weighted_index(&[]), Err(RngError::NoPositiveWeights));
        assert_eq!(
            rng.weighted_index(&[0, 0]),
            Err(RngError::NoPositiveWeights)
        );

        let mut rng = Rng::from_seed_u64(0);
        assert_eq!(
            rng.weighted_index(&[usize::MAX, 1]),
            Err(RngError::SampleSpaceOverflow)
        );
    }

    #[test]
    fn weighted_index_maps_draws_to_weight_boundaries() {
        let weights = [2, 3, 5];
        for (draw, expected_index) in [(0, 0), (1, 0), (2, 1), (4, 1), (5, 2), (9, 2)] {
            let mut rng = Rng::from_source(FixedSource::new([draw]));
            assert_eq!(rng.weighted_index(&weights), Ok(expected_index));
        }
    }

    #[test]
    fn weighted_index_with_one_positive_weight_selects_it() {
        for draw in [0, 1, 6] {
            let mut rng = Rng::from_source(FixedSource::new([draw]));
            assert_eq!(rng.weighted_index(&[0, 7, 0]), Ok(1));
        }
    }
}
