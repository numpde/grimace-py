//! Private bounded load-word experiment for a single fixed boundary context.
//!
//! A word records loads at cuts, including both endpoint cuts. Its quotient is
//! mutual domination under repetitions, not equality of attainable schedules.

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct LoadWord(pub(super) Vec<usize>);

impl LoadWord {
    pub(super) fn new(loads: &[usize], capacity: usize) -> Option<Self> {
        assert!(!loads.is_empty(), "a load word needs an endpoint cut");
        let mut word = Self(Vec::new());
        for &load in loads {
            if load > capacity {
                return None;
            }
            word.extend(load);
        }
        Some(word)
    }

    fn extend(&mut self, load: usize) {
        if self.0.last() == Some(&load) {
            return;
        }
        self.0.push(load);
        // An interior segment bounded by x,y in which every load lies between
        // x and y can be stretched away in either repetition direction.
        loop {
            let end = self.0.len() - 1;
            let removable = (0..end.saturating_sub(1)).rev().find(|&start| {
                let lo = self.0[start].min(load);
                let hi = self.0[start].max(load);
                self.0[start + 1..end].iter().all(|&v| lo <= v && v <= hi)
            });
            match removable {
                Some(start) => {
                    self.0.drain(start + 1..end);
                }
                None => break,
            }
        }
    }

    pub(super) fn dominates(&self, other: &Self) -> bool {
        let (a, b) = (&self.0, &other.0);
        let mut reachable = vec![vec![false; b.len()]; a.len()];
        reachable[0][0] = a[0] <= b[0];
        for i in 0..a.len() {
            for j in 0..b.len() {
                if a[i] > b[j] || (i == 0 && j == 0) {
                    continue;
                }
                reachable[i][j] = (i > 0 && reachable[i - 1][j])
                    || (j > 0 && reachable[i][j - 1])
                    || (i > 0 && j > 0 && reachable[i - 1][j - 1]);
            }
        }
        reachable[a.len() - 1][b.len() - 1]
    }
}

// Profiles at one cell share both endpoint cuts and every possible suffix.
// Pruning across cells would confuse different remaining interleavings.
fn insert_antichain(profiles: &mut Vec<LoadWord>, candidate: LoadWord) {
    if profiles.iter().any(|p| p.dominates(&candidate)) {
        return;
    }
    profiles.retain(|p| !candidate.dominates(p));
    profiles.push(candidate);
}

pub(super) fn merge(a: &LoadWord, b: &LoadWord, capacity: usize) -> Vec<LoadWord> {
    // If both words start and end at their global minima, serial traversal has
    // profile [c, p, c], where c is the sum of minima and p is the larger
    // one-word peak plus the other's minimum. Every H/V merge stays at least
    // c and reaches at least p; hold the serial word at c until that peak,
    // run it while the other merge stutters there, then hold it at c again.
    // This is a resource-word domination proof only: traversal legality and
    // writer effects need a separate shared control/serialization argument.
    let rows = a.0.len();
    let cols = b.0.len();
    let mut cells = vec![vec![Vec::new(); cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            let Some(load) = a.0[i].checked_add(b.0[j]) else {
                continue;
            };
            if load > capacity {
                continue;
            }
            if i == 0 && j == 0 {
                cells[i][j].push(LoadWord(vec![load]));
                continue;
            }
            let mut next = Vec::new();
            for previous in (i > 0)
                .then(|| &cells[i - 1][j])
                .into_iter()
                .chain((j > 0).then(|| &cells[i][j - 1]))
            {
                for prefix in previous {
                    let mut candidate = prefix.clone();
                    candidate.extend(load);
                    insert_antichain(&mut next, candidate);
                }
            }
            cells[i][j] = next;
        }
    }
    cells.pop().unwrap().pop().unwrap()
}

#[cfg(test)]
mod tests {
    use super::{merge, LoadWord};

    fn word(loads: &[usize], k: usize) -> LoadWord {
        LoadWord::new(loads, k).unwrap()
    }

    // Independent raw H/V schedules. This routine does not reduce or compare
    // words and therefore checks the semantic peak answer, not representation.
    fn raw_peak(a: &[usize], b: &[usize], outside: &[usize]) -> Option<usize> {
        fn walk(words: &[&[usize]; 3], at: [usize; 3], peak: usize, best: &mut Option<usize>) {
            if at.iter().enumerate().all(|(d, &i)| i + 1 == words[d].len()) {
                *best = Some(best.map_or(peak, |old| old.min(peak)));
                return;
            }
            for d in 0..3 {
                if at[d] + 1 == words[d].len() {
                    continue;
                }
                let mut next = at;
                next[d] += 1;
                let load = (0..3).map(|n| words[n][next[n]]).sum();
                walk(words, next, peak.max(load), best);
            }
        }
        let words = [a, b, outside];
        let start = a[0] + b[0] + outside[0];
        let mut best = None;
        walk(&words, [0; 3], start, &mut best);
        best
    }

    fn profile_peak(profiles: &[LoadWord], outside: &[usize]) -> Option<usize> {
        profiles
            .iter()
            .filter_map(|p| raw_peak(&p.0, &[0], outside))
            .min()
    }

    fn one_nesting(a: &[usize], b: &[usize], capacity: usize) -> Vec<LoadWord> {
        fn insert(inner: &[usize], outer: &[usize], capacity: usize) -> Vec<LoadWord> {
            (0..outer.len())
                .map(|split| {
                    let mut loads = vec![outer[0] + inner[0]];
                    loads.extend(outer[1..split + 1].iter().map(|&v| v + inner[0]));
                    loads.extend(inner[1..].iter().map(|&v| v + outer[split]));
                    loads.extend(
                        outer[split + 1..]
                            .iter()
                            .map(|&v| v + inner[inner.len() - 1]),
                    );
                    word(&loads, capacity)
                })
                .collect()
        }
        let mut profiles = insert(b, a, capacity);
        profiles.extend(insert(a, b, capacity));
        profiles
    }

    fn free_path(a: &[usize], b: &[usize], steps: &[usize]) -> Vec<usize> {
        let mut at = [0, 0];
        let words = [a, b];
        let mut loads = vec![a[0] + b[0]];
        for &side in steps {
            assert!(side < 2 && at[side] + 1 < words[side].len());
            at[side] += 1;
            loads.push(a[at[0]] + b[at[1]]);
        }
        assert_eq!(at, [a.len() - 1, b.len() - 1]);
        loads
    }

    #[test]
    fn minimum_ended_words_have_a_dominating_serial_profile() {
        let cases: &[(&[usize], &[usize], &[usize])] = &[
            (&[0, 2, 0], &[0, 1, 0], &[0, 2, 0]),
            (&[2, 3, 2], &[1, 4, 1], &[3, 6, 3]),
            (&[1, 3, 1, 2, 1], &[2, 3, 2], &[3, 5, 3]),
        ];
        for &(a, b, canonical) in cases {
            let profiles = merge(&word(a, 8), &word(b, 8), 8);
            assert_eq!(profiles, vec![word(canonical, 8)], "a={a:?}, b={b:?}");
        }
    }

    #[test]
    fn equal_endpoints_do_not_make_one_nesting_complete() {
        let cases: &[(&[usize], &[usize], &[usize], &[usize])] = &[
            (
                &[1, 2, 0, 1],
                &[1, 0, 2, 1],
                &[1, 0, 0, 1, 1, 0],
                &[2, 1, 2, 0, 2, 1, 2],
            ),
            (
                &[1, 2, 1, 0, 1],
                &[1, 0, 1, 2, 1],
                &[1, 0, 0, 0, 1, 1, 1, 0],
                &[2, 1, 2, 1, 0, 1, 2, 1, 2],
            ),
        ];
        for &(a, b, steps, expected) in cases {
            let free_loads = free_path(a, b, steps);
            assert_eq!(free_loads, expected);
            let free = word(&free_loads, 4);
            assert_eq!(free.0.iter().copied().max(), Some(2));
            let profiles = merge(&word(a, 4), &word(b, 4), 4);
            assert!(profiles.iter().any(|p| p.dominates(&free)));
            let nested = one_nesting(a, b, 4);
            assert!(nested
                .iter()
                .all(|p| p.0.iter().copied().max().unwrap() >= 3));
            assert!(nested.iter().all(|p| !p.dominates(&free)));
            assert!(nested.iter().any(|p| p.0.iter().copied().max() == Some(3)));
        }
    }

    #[test]
    fn reduction_preserves_endpoints_peak_and_small_word_bound() {
        for k in 0..=3 {
            for len in 1..=7 {
                let count = (k + 1usize).pow(len);
                for mut code in 0..count {
                    let loads: Vec<_> = (0..len)
                        .map(|_| {
                            let v = code % (k + 1);
                            code /= k + 1;
                            v
                        })
                        .collect();
                    let reduced = word(&loads, k);
                    assert_eq!(reduced.0.first(), loads.first());
                    assert_eq!(reduced.0.last(), loads.last());
                    assert_eq!(reduced.0.iter().max(), loads.iter().max());
                    assert!(reduced.0.len() <= 2 * k + 1);
                    let raw = LoadWord(loads);
                    assert!(raw.dominates(&reduced));
                    assert!(reduced.dominates(&raw));
                }
            }
        }
    }

    #[test]
    fn merge_matches_raw_schedule_peaks_in_small_contexts() {
        let cases: &[&[usize]] = &[&[0], &[1], &[0, 1], &[1, 0], &[1, 0, 1], &[2, 0, 2]];
        let contexts: &[&[usize]] = &[&[0], &[1], &[0, 1, 0], &[1, 0, 1]];
        for &a in cases {
            for &b in cases {
                let profiles = merge(&word(a, 4), &word(b, 4), 4);
                for &outside in contexts {
                    assert_eq!(
                        profile_peak(&profiles, outside),
                        raw_peak(a, b, outside),
                        "a={a:?}, b={b:?}, outside={outside:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn capacity_and_association_are_quotient_properties() {
        let z = word(&[0], 3);
        assert_eq!(merge(&word(&[2], 3), &z, 3), vec![word(&[2], 3)]);
        assert!(merge(&word(&[2], 3), &word(&[2], 3), 3).is_empty());
        assert!(LoadWord::new(&[0, 4], 3).is_none());
        let cases = [&[0, 1, 0][..], &[1, 0][..], &[0, 2][..]];
        for &a in &cases {
            for &b in &cases {
                for &c in &cases {
                    let (a, b, c) = (word(a, 4), word(b, 4), word(c, 4));
                    let left: Vec<_> = merge(&a, &b, 4)
                        .iter()
                        .flat_map(|p| merge(p, &c, 4))
                        .collect();
                    let right: Vec<_> = merge(&b, &c, 4)
                        .iter()
                        .flat_map(|p| merge(&a, p, 4))
                        .collect();
                    for p in &left {
                        assert!(right.iter().any(|q| q.dominates(p)));
                    }
                    for p in &right {
                        assert!(left.iter().any(|q| q.dominates(p)));
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_outside_load_shifts_capacity_without_changing_profiles() {
        let a = word(&[0, 2, 0], 4);
        let b = word(&[1, 0, 1], 4);
        for outside in 0..=2 {
            for capacity in outside..=4 {
                let base = merge(&a, &b, capacity - outside);
                let shifted_a = word(&a.0.iter().map(|&v| v + outside).collect::<Vec<_>>(), 4);
                let shifted = merge(&shifted_a, &b, capacity);
                let expected: Vec<_> = base
                    .iter()
                    .map(|p| {
                        word(
                            &p.0.iter().map(|&v| v + outside).collect::<Vec<_>>(),
                            capacity,
                        )
                    })
                    .collect();
                assert_eq!(shifted, expected, "outside={outside}, capacity={capacity}");
            }
        }
    }

    #[test]
    fn merge_keeps_only_nondominated_profiles_at_one_boundary() {
        let profiles = merge(&word(&[2, 0, 2], 4), &word(&[0, 2, 0], 4), 4);
        assert!(!profiles.is_empty());
        for (i, profile) in profiles.iter().enumerate() {
            assert_eq!(profile.0.first(), Some(&2));
            assert_eq!(profile.0.last(), Some(&2));
            assert!(profile.0.iter().all(|&v| v <= 4));
            for (j, other) in profiles.iter().enumerate() {
                if i != j {
                    assert!(!profile.dominates(other));
                }
            }
        }
    }

    #[test]
    fn equal_peaks_need_distinct_valleys() {
        let valley = word(&[2, 0, 2], 4);
        let flat = word(&[2], 4);
        assert!(!flat.dominates(&valley));
        assert!(valley.dominates(&flat));
        let outside = [0, 2, 0];
        assert_eq!(raw_peak(&valley.0, &[0], &outside), Some(2));
        assert_eq!(raw_peak(&flat.0, &[0], &outside), Some(4));
    }
}
