//! Translate the console's DispatchX thread dimensions to host groups.

pub(crate) const THREAD_DIMENSIONS: u32 = 0x20;

/// Return groups and an optional guard limit for the shader's last X group.
/// Do not overdispatch if the translated shader cannot represent a tail.
pub(crate) fn plan(
    count: [u32; 3], size: Option<[u32; 3]>, flags: u32, guard: bool,
) -> Option<([u32; 3], u32)> {
    if flags & THREAD_DIMENSIONS == 0 {
        return Some((count, 0));
    }
    if count.contains(&0) {
        return Some(([0; 3], 0));
    }
    let size = size.filter(|s| !s.contains(&0))?;
    if !count[1].is_multiple_of(size[1]) || !count[2].is_multiple_of(size[2]) {
        return None;
    }
    let partial = !count[0].is_multiple_of(size[0]);
    if partial && !guard {
        return None;
    }
    let groups = std::array::from_fn(|i| count[i].div_ceil(size[i]));
    Some((groups, if partial { count[0] } else { 0 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_counts_use_ceil_but_require_a_guard_for_extra_lanes() {
        let f = THREAD_DIMENSIONS;
        let s = Some([64, 1, 1]);
        assert_eq!(plan([8847, 1, 1], s, f, true), Some(([139, 1, 1], 8847)));
        assert_eq!(plan([74, 1, 1], s, f, true), Some(([2, 1, 1], 74)));
        assert_eq!(plan([74, 1, 1], s, f, false), None);
        assert_eq!(plan([32, 1, 1], Some([32, 1, 1]), f, false), Some(([1; 3], 0)));
        assert_eq!(plan([0, 12, 9], Some([64, 4, 4]), f, false), Some(([0; 3], 0)));
        assert_eq!(plan([64, 5, 1], Some([64, 4, 1]), f, true), None);
        assert_eq!(plan([74, 1, 1], None, f, true), None);
        assert_eq!(plan([74, 1, 1], None, 0, false), Some(([74, 1, 1], 0)));
    }
}
