//! Opt-in, thread-local operation counts for bounded writer work probes.

use std::cell::RefCell;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Counts {
    pub(crate) paged_reads: usize,
    pub(crate) factor_revisions: usize,
    pub(crate) search_nodes: usize,
    pub(crate) grouped_branches: usize,
    pub(crate) scalar_branches: usize,
    pub(crate) sibling_offered: usize,
    // Counts attempted restrictions, including contradictions.
    pub(crate) sibling_proposals: usize,
    pub(crate) sibling_propagated: usize,
    pub(crate) children_explored: usize,
    pub(crate) candidate_attempts: usize,
    pub(crate) candidate_rejections: usize,
}

thread_local! {
    static ACTIVE: RefCell<Option<Counts>> = const { RefCell::new(None) };
}

pub(crate) fn begin() { ACTIVE.with(|active| *active.borrow_mut() = Some(Counts::default())); }
pub(crate) fn finish() -> Counts { ACTIVE.with(|active| active.borrow_mut().take().unwrap()) }
pub(crate) fn record(update: impl FnOnce(&mut Counts)) {
    ACTIVE.with(|active| {
        if let Some(counts) = active.borrow_mut().as_mut() { update(counts); }
    });
}
