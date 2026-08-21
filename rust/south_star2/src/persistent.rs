//! Persistent indexed storage for sparse writer-state forks.

use std::fmt;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

const VALUES_PER_PAGE: usize = 64;
const CHILDREN_PER_NODE: usize = 32;

#[derive(Clone)]
enum Node<T> {
    Branch(Box<[Arc<Node<T>>]>),
    Leaf(Box<[T]>),
}

#[cfg(test)]
#[derive(Debug, Default)]
struct CopyCounters {
    index_nodes: AtomicUsize,
    value_pages: AtomicUsize,
}

#[derive(Clone)]
pub(crate) struct PagedStore<T> {
    len: usize,
    height: usize,
    root: Option<Arc<Node<T>>>,
    #[cfg(test)]
    copy_counters: Arc<CopyCounters>,
}

impl<T: Clone> PagedStore<T> {
    pub(crate) fn from_values(values: impl IntoIterator<Item = T>) -> Self {
        let values = values.into_iter().collect::<Vec<_>>();
        let len = values.len();
        let mut nodes = values
            .chunks(VALUES_PER_PAGE)
            .map(|page| Arc::new(Node::Leaf(page.to_vec().into_boxed_slice())))
            .collect::<Vec<_>>();
        let mut height = 0;

        while nodes.len() > 1 {
            nodes = nodes
                .chunks(CHILDREN_PER_NODE)
                .map(|children| Arc::new(Node::Branch(children.to_vec().into_boxed_slice())))
                .collect();
            height += 1;
        }

        Self {
            len,
            height,
            root: nodes.pop(),
            #[cfg(test)]
            copy_counters: Arc::new(CopyCounters::default()),
        }
    }

    pub(crate) fn filled(len: usize, value: T) -> Self {
        Self::from_values(std::iter::repeat_n(value, len))
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let page = index / VALUES_PER_PAGE;
        let offset = index % VALUES_PER_PAGE;
        Some(get_from_node(
            self.root
                .as_ref()
                .expect("a nonempty store must have a root"),
            self.height,
            page,
            offset,
        ))
    }

    pub(crate) fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        let page = index / VALUES_PER_PAGE;
        let offset = index % VALUES_PER_PAGE;
        Some(get_mut_from_node(
            self.root
                .as_mut()
                .expect("a nonempty store must have a root"),
            self.height,
            page,
            offset,
            #[cfg(test)]
            &self.copy_counters,
        ))
    }

    #[cfg(test)]
    pub(crate) fn reset_copy_counts(&self) {
        self.copy_counters.index_nodes.store(0, Ordering::Relaxed);
        self.copy_counters.value_pages.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn copy_counts(&self) -> (usize, usize) {
        (
            self.copy_counters.index_nodes.load(Ordering::Relaxed),
            self.copy_counters.value_pages.load(Ordering::Relaxed),
        )
    }

    #[cfg(test)]
    pub(crate) fn shares_value_page_with(&self, other: &Self, index: usize) -> bool {
        assert!(index < self.len && index < other.len);
        let page = index / VALUES_PER_PAGE;
        Arc::ptr_eq(
            value_page(
                self.root.as_ref().expect("nonempty store must have a root"),
                self.height,
                page,
            ),
            value_page(
                other
                    .root
                    .as_ref()
                    .expect("nonempty store must have a root"),
                other.height,
                page,
            ),
        )
    }
}

impl<T: Clone> Index<usize> for PagedStore<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("paged-store index must be in range")
    }
}

impl<T: Clone> IndexMut<usize> for PagedStore<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("paged-store index must be in range")
    }
}

impl<T: Clone + PartialEq> PartialEq for PagedStore<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && (0..self.len).all(|index| self[index] == other[index])
    }
}

impl<T: Clone + Eq> Eq for PagedStore<T> {}

impl<T: Clone + fmt::Debug> fmt::Debug for PagedStore<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_list()
            .entries((0..self.len).map(|index| &self[index]))
            .finish()
    }
}

fn get_from_node<T>(node: &Arc<Node<T>>, height: usize, page: usize, offset: usize) -> &T {
    match node.as_ref() {
        Node::Leaf(values) => {
            assert_eq!(height, 0);
            &values[offset]
        }
        Node::Branch(children) => {
            assert!(height > 0);
            let pages_per_child = CHILDREN_PER_NODE.pow((height - 1) as u32);
            get_from_node(
                &children[page / pages_per_child],
                height - 1,
                page % pages_per_child,
                offset,
            )
        }
    }
}

fn get_mut_from_node<'a, T: Clone>(
    node: &'a mut Arc<Node<T>>,
    height: usize,
    page: usize,
    offset: usize,
    #[cfg(test)] copy_counters: &CopyCounters,
) -> &'a mut T {
    #[cfg(test)]
    let was_shared = Arc::strong_count(node) > 1;
    let node = Arc::make_mut(node);

    match node {
        Node::Leaf(values) => {
            assert_eq!(height, 0);
            #[cfg(test)]
            if was_shared {
                copy_counters.value_pages.fetch_add(1, Ordering::Relaxed);
            }
            &mut values[offset]
        }
        Node::Branch(children) => {
            assert!(height > 0);
            #[cfg(test)]
            if was_shared {
                copy_counters.index_nodes.fetch_add(1, Ordering::Relaxed);
            }
            let pages_per_child = CHILDREN_PER_NODE.pow((height - 1) as u32);
            get_mut_from_node(
                &mut children[page / pages_per_child],
                height - 1,
                page % pages_per_child,
                offset,
                #[cfg(test)]
                copy_counters,
            )
        }
    }
}

#[cfg(test)]
fn value_page<T>(node: &Arc<Node<T>>, height: usize, page: usize) -> &Arc<Node<T>> {
    if height == 0 {
        return node;
    }
    let Node::Branch(children) = node.as_ref() else {
        unreachable!("a positive-height node must be a branch");
    };
    let pages_per_child = CHILDREN_PER_NODE.pow((height - 1) as u32);
    value_page(
        &children[page / pages_per_child],
        height - 1,
        page % pages_per_child,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_forks_copy_only_the_written_paths_and_pages() {
        let source = PagedStore::from_values(0..10_000);
        let mut left = source.clone();
        let mut right = source.clone();
        source.reset_copy_counts();

        left[3] = -3;
        right[9_999] = -9_999;

        assert_eq!(source[3], 3);
        assert_eq!(source[9_999], 9_999);
        assert_eq!(left[3], -3);
        assert_eq!(right[9_999], -9_999);
        assert!(left.shares_value_page_with(&source, 9_999));
        assert!(right.shares_value_page_with(&source, 3));
        assert_eq!(source.copy_counts(), (4, 2));
    }

    #[test]
    fn empty_and_partial_pages_preserve_indexing() {
        let empty = PagedStore::<usize>::from_values([]);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.get(0), None);

        let mut values = PagedStore::filled(65, 7);
        values[64] = 11;
        assert_eq!(values[0], 7);
        assert_eq!(values[64], 11);
    }

    #[test]
    fn radix_boundaries_match_dense_forks() {
        for len in [63, 64, 65, 2_048, 2_049, 65_536, 65_537] {
            let source = PagedStore::from_values(0..len);
            let mut fork = source.clone();
            let mut expected = (0..len).collect::<Vec<_>>();
            let writes = [0, len / 2, len - 1, 0];

            for (step, index) in writes.into_iter().enumerate() {
                let value = len + step;
                fork[index] = value;
                expected[index] = value;
            }

            assert!(
                (0..len).all(|index| source[index] == index),
                "source length {len}"
            );
            assert!(
                (0..len).all(|index| fork[index] == expected[index]),
                "fork length {len}"
            );
        }
    }
}
