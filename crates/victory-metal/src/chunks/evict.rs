pub(crate) fn remove_then_release<T>(
    buffers: Vec<T>,
    remove: impl FnOnce(&[T]),
    commit: impl FnOnce(),
) -> usize {
    let count = buffers.len();
    if count != 0 {
        remove(&buffers);
        commit();
        drop(buffers);
    }
    count
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use super::*;

    struct Wrapper<'a>(&'a RefCell<Vec<&'static str>>);
    impl Drop for Wrapper<'_> {
        fn drop(&mut self) {
            self.0.borrow_mut().push("release");
        }
    }

    #[test]
    fn commits_removal_before_releasing_any_wrapper() {
        let log = RefCell::new(Vec::new());
        let n = remove_then_release(
            vec![Wrapper(&log), Wrapper(&log)],
            |items| {
                assert_eq!(items.len(), 2);
                log.borrow_mut().push("remove");
            },
            || log.borrow_mut().push("commit"),
        );
        log.borrow_mut().extend(["replace mapping", "wrap"]);
        assert_eq!(n, 2);
        assert_eq!(*log.borrow(), ["remove", "commit", "release", "release", "replace mapping", "wrap"]);
    }

    #[test]
    fn empty_eviction_does_not_commit() {
        assert_eq!(remove_then_release(Vec::<()>::new(), |_| panic!("removed empty set"), || panic!("committed empty set")), 0);
    }
}
