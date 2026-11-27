//! Loops each lane leaves on its own, for `lower::lane_regions`.
//!
//! A lane branch on a cycle (`v_cmpx` + `s_andn2_b64` of the lanes left + `s_cbranch_scc1` to
//! a latch that takes EXEC back to them) runs every block of the cycle as many times as the
//! lane stays, which differs from lane to lane. The branch's region, the blocks reachable from
//! it that do not post-dominate it, leaves out the header (it post-dominates itself), so the
//! loop's blocks counted as code every lane runs: the two host waves of a console wave then
//! met there (a half-wave sync before an LDS access, an exchange) a different number of times
//! when the lanes that stay longer sat in one half, fell out of step, and one waited out the
//! poll limit (`a1bf269d5b19f025`'s counting loops with an LDS add; the one half-sync timeout
//! of the M3a run). The cycle's blocks are in the region too, the branch's own block with them.
//! `VICTORY_WAVE_LOOP_REGIONS=0`: not, as before (78 of the F85 programs change).

/// Adds to `region` every block on a cycle through the lane branch closing block `d`.
pub(super) fn add_cycle(d: usize, succ: &[Vec<usize>], region: &mut [bool]) {
    if std::env::var_os("VICTORY_WAVE_LOOP_REGIONS").is_some_and(|v| v == "0") {
        return;
    }
    let n = succ.len();
    let walk = |from: &[usize], next: &dyn Fn(usize) -> Vec<usize>| {
        let mut got = vec![false; n];
        let mut stack = from.to_vec();
        while let Some(b) = stack.pop() {
            if !got[b] {
                got[b] = true;
                stack.extend(next(b));
            }
        }
        got
    };
    let fwd = walk(&succ[d], &|b| succ[b].clone());
    if !fwd[d] {
        return;
    }
    let back = walk(&[d], &|b| {
        (0..n).filter(|&p| succ[p].contains(&b)).collect()
    });
    for b in (0..n).filter(|&b| fwd[b] && back[b]) {
        region[b] = true;
    }
}
