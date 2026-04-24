pub(crate) fn ring_label_text(label: usize) -> String {
    if label < 10 {
        label.to_string()
    } else if label < 100 {
        format!("%{label}")
    } else {
        format!("%({label})")
    }
}

pub(crate) fn add_pending<T: Ord>(
    pending: &mut Vec<(usize, Vec<T>)>,
    target_atom: usize,
    entry: T,
) {
    let current = match pending.binary_search_by_key(&target_atom, |(atom_idx, _)| *atom_idx) {
        Ok(offset) => &mut pending[offset].1,
        Err(offset) => {
            pending.insert(offset, (target_atom, Vec::new()));
            &mut pending[offset].1
        }
    };
    let insert_at = current
        .binary_search(&entry)
        .unwrap_or_else(|offset| offset);
    current.insert(insert_at, entry);
}

pub(crate) fn take_pending_for_atom<T>(
    pending: &mut Vec<(usize, Vec<T>)>,
    atom_idx: usize,
) -> Vec<T> {
    match pending.binary_search_by_key(&atom_idx, |(candidate_idx, _)| *candidate_idx) {
        Ok(offset) => pending.remove(offset).1,
        Err(_) => Vec::new(),
    }
}
