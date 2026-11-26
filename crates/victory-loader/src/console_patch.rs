//! Patch-once for console-only instructions (T8 follow-up). The first time a site traps it is
//! emulated as before, then rewritten so it never traps again:
//! - `monitorx` → NOPs, `mwaitx` → `pause`, `rdpru` → `rdtsc`, `clwb` → NOPs (in place);
//! - `clzero` and `rdpid` → a short jump into the nearest function padding (a run of `int3`),
//!   which jumps on to a trampoline doing the same work and jumping back;
//! - SSE4a `insertq` / `extrq` immediate forms → the same padding/trampoline route, with the
//!   bit field done by ordinary SSE2 + general-register instructions.
//!
//! Other threads may run the site while it changes: its first byte becomes `int3` before
//! the rest is written, and a breakpoint or a stale trap at a site in [`SITES`] just runs the
//! instruction again. `VICTORY_INSN_PATCH=0` keeps the old trap-every-time emulation.
//!
//! The machine-code generation below the Windows glue is pure: inputs are instruction bytes,
//! register numbers or addresses, outputs are `Vec<u8>`, with no process state, so the F3
//! harness can include this file and run the encoders natively.

#[cfg(windows)]
use std::collections::HashSet;
#[cfg(windows)]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
#[cfg(windows)]
use std::sync::{Mutex, OnceLock};

// =====================================================================================
// Pure machine-code generation (no Windows calls, no process state).
// =====================================================================================

/// A register number: 0..=15 (rax..r15 / xmm0..xmm15).
type Reg = u8;

/// ModRM byte with both operands registers (mod = 3).
fn modrm_reg(reg: Reg, rm: Reg) -> u8 {
    0xC0 | ((reg & 7) << 3) | (rm & 7)
}

/// NOPs filling `len` bytes (clwb / monitorx replacement).
pub fn fill_nop(len: usize) -> Vec<u8> {
    vec![0x90; len]
}

/// `pause` (F3 90) in the first two bytes, NOPs after (mwaitx replacement).
pub fn fill_pause(len: usize) -> Vec<u8> {
    let mut b = fill_nop(len);
    if len >= 2 {
        b[..2].copy_from_slice(&[0xF3, 0x90]);
    }
    b
}

/// `rdtsc` (0F 31) in the first two bytes, NOPs after (rdpru replacement).
pub fn fill_rdtsc(len: usize) -> Vec<u8> {
    let mut b = fill_nop(len);
    if len >= 2 {
        b[..2].copy_from_slice(&[0x0F, 0x31]);
    }
    b
}

/// rel32 displacement from the end of the relocation instruction at `from_end` to `to`.
pub fn rel32(from_end: usize, to: usize) -> Option<[u8; 4]> {
    let d = to as i64 - from_end as i64;
    i32::try_from(d).ok().map(|d| d.to_le_bytes())
}

/// The native clzero-loop trampoline body: zero the 64-byte line at rax, advance rax by
/// 0x40, loop back to the head while rax < end. `cmp` is the decoded loop's 3-byte
/// `cmp rax, end_reg`.
pub fn clzero_loop_body(cmp: &[u8]) -> Vec<u8> {
    let mut t = vec![0x51, 0x48, 0x89, 0xC1, 0x48, 0x83, 0xE1, 0xC0]; // push rcx; mov rcx, rax; and rcx, -64
    t.extend_from_slice(&[0x48, 0xC7, 0x01, 0, 0, 0, 0]); // mov qword [rcx], 0
    for off in (8u8..64).step_by(8) {
        t.extend_from_slice(&[0x48, 0xC7, 0x41, off, 0, 0, 0, 0]); // mov qword [rcx+off], 0
    }
    t.extend_from_slice(&[0x59, 0x48, 0x83, 0xC0, 0x40]); // pop rcx; add rax, 0x40
    t.extend_from_slice(cmp); // cmp rax, <end>
    let back = -((t.len() + 2) as i64);
    t.extend_from_slice(&[0x72, back as i8 as u8]); // jb loop
    t
}

/// The lone-clzero trampoline body. `a32`: the site had a 0x67 address-size prefix, so the
/// line is at EAX (zero-extended); otherwise at RAX.
pub fn clzero_body(a32: bool) -> Vec<u8> {
    let mut t = vec![0x9C, 0x51]; // pushfq; push rcx
    t.extend_from_slice(if a32 { &[0x89, 0xC1][..] } else { &[0x48, 0x89, 0xC1][..] }); // mov rcx, rax
    t.extend_from_slice(&[0x48, 0x83, 0xE1, 0xC0]); // and rcx, -64
    t.extend_from_slice(&[0x48, 0xC7, 0x01, 0, 0, 0, 0]); // mov qword [rcx], 0
    for off in (8u8..64).step_by(8) {
        t.extend_from_slice(&[0x48, 0xC7, 0x41, off, 0, 0, 0, 0]); // mov qword [rcx+off], 0
    }
    t.extend_from_slice(&[0x59, 0x9D]); // pop rcx; popfq
    t
}

/// The `rdpid` trampoline body: `mov r<reg>, 0` (processor number 0 is a valid answer).
pub fn rdpid_body(r: Reg) -> Vec<u8> {
    if r < 8 {
        vec![0xB8 + r, 0, 0, 0, 0]
    } else {
        vec![0x41, 0xB8 + (r - 8), 0, 0, 0, 0]
    }
}

/// Site replacement: `jmp rel32 to`, NOP-padded to `len` bytes (len ≥ 5).
pub fn jmp_rel_site(len: usize, site: usize, to: usize) -> Vec<u8> {
    let mut s = fill_nop(len);
    s[0] = 0xE9;
    if let Some(rel) = rel32(site + 5, to) {
        s[1..5].copy_from_slice(&rel);
    }
    s
}

/// Site replacement: `jmp short pad`, NOP-padded to `len` bytes.
pub fn jmp_short_site(len: usize, site: usize, pad: usize) -> Vec<u8> {
    let mut s = fill_nop(len);
    s[0] = 0xEB;
    s[1] = (pad as i64 - (site + 2) as i64) as i8 as u8;
    s
}

/// Padding slot replacement: `jmp rel32 tramp`.
pub fn pad_jump(pad: usize, tramp: usize) -> Vec<u8> {
    let mut p = vec![0xE9];
    if let Some(rel) = rel32(pad + 5, tramp) {
        p.extend_from_slice(&rel);
    }
    p
}

/// Trampoline image: `body` followed by `jmp rel32 site_end`; None if out of rel32 range.
pub fn trampoline_image(body: &[u8], tramp: usize, site_end: usize) -> Option<Vec<u8>> {
    let back = rel32(tramp + body.len() + 5, site_end)?;
    let mut t = body.to_vec();
    t.push(0xE9);
    t.extend_from_slice(&back);
    Some(t)
}

/// `movq r64, xmm`: 66 REX.W 0F 7E /r (ModRM reg = xmm, rm = gpr). REX.W is required:
/// without it the same opcode is the 32-bit `movd`; the REX sits after the 66 prefix.
fn movq_gpr_from_xmm(gpr: Reg, xmm: Reg, out: &mut Vec<u8>) {
    out.push(0x66);
    out.push(0x48 | ((xmm & 8) >> 1) | (gpr & 8));
    out.extend_from_slice(&[0x0F, 0x7E, modrm_reg(xmm, gpr)]);
}

/// `movq xmm, r64`: 66 REX.W 0F 6E /r (ModRM reg = xmm, rm = gpr); zeros the xmm high qword.
fn movq_xmm_from_gpr(xmm: Reg, gpr: Reg, out: &mut Vec<u8>) {
    out.push(0x66);
    out.push(0x48 | ((xmm & 8) >> 1) | (gpr & 8));
    out.extend_from_slice(&[0x0F, 0x6E, modrm_reg(xmm, gpr)]);
}

/// `movabs r<gpr>, imm64` (REX.W).
fn movabs(gpr: Reg, imm: u64, out: &mut Vec<u8>) {
    out.push(0x48 | (gpr >> 3));
    out.push(0xB8 + (gpr & 7));
    out.extend_from_slice(&imm.to_le_bytes());
}

/// The ones-mask for a SSE4a field length: 0 (and 64) mean all 64 bits.
fn field_ones(len: u8) -> u64 {
    match len & 63 {
        0 => u64::MAX,
        n => (1u64 << n) - 1,
    }
}

/// SSE4a `extrq xmm, imm8 len, imm8 idx` trampoline body (immediate form only): the low
/// quadword is shifted down by `idx` and masked to `len` bits; the high quadword is zeroed,
/// as the existing emulation does. Scratch: rax/rcx, spilled with push/pop; flags preserved
/// via pushfq/popfq.
pub fn extrq_body(dst: Reg, len: u8, idx: u8) -> Vec<u8> {
    let idx = idx & 63;
    let ones = field_ones(len);
    let mut t = Vec::new();
    t.push(0x9C); // pushfq
    t.push(0x50); // push rax
    t.push(0x51); // push rcx
    movq_gpr_from_xmm(0, dst, &mut t); // movq rax, xmm_dst
    if idx > 0 {
        t.extend_from_slice(&[0x48, 0xC1, 0xE8, idx]); // shr rax, idx
    }
    if ones != u64::MAX {
        movabs(1, ones, &mut t); // movabs rcx, ones
        t.extend_from_slice(&[0x48, 0x21, 0xC8]); // and rax, rcx
    }
    movq_xmm_from_gpr(dst, 0, &mut t); // movq xmm_dst, rax (zeros the high qword)
    t.push(0x59); // pop rcx
    t.push(0x58); // pop rax
    t.push(0x9D); // popfq
    t
}

/// SSE4a `insertq xmm_dst, xmm_src, imm8 len, imm8 idx` trampoline body (immediate form
/// only): the low `len` bits of src are placed at bit `idx` of dst's low quadword; the high
/// quadword is zeroed. Scratch: rax/rcx/rdx, spilled; flags preserved.
pub fn insertq_body(dst: Reg, src: Reg, len: u8, idx: u8) -> Vec<u8> {
    let idx = idx & 63;
    let ones = field_ones(len);
    let mask = ones.checked_shl(idx as u32).unwrap_or(0);
    let not_mask = !mask;
    let mut t = Vec::new();
    t.push(0x9C); // pushfq
    t.push(0x50); // push rax
    t.push(0x51); // push rcx
    t.push(0x52); // push rdx
    movq_gpr_from_xmm(0, dst, &mut t); // movq rax, xmm_dst
    movq_gpr_from_xmm(2, src, &mut t); // movq rdx, xmm_src
    movabs(1, not_mask, &mut t);
    t.extend_from_slice(&[0x48, 0x21, 0xC8]); // and rax, rcx
    movabs(1, ones, &mut t);
    t.extend_from_slice(&[0x48, 0x21, 0xCA]); // and rdx, rcx
    if idx > 0 {
        t.extend_from_slice(&[0x48, 0xC1, 0xE2, idx]); // shl rdx, idx
    }
    movabs(1, mask, &mut t);
    t.extend_from_slice(&[0x48, 0x21, 0xCA]); // and rdx, rcx
    t.extend_from_slice(&[0x48, 0x09, 0xD0]); // or rax, rdx
    movq_xmm_from_gpr(dst, 0, &mut t); // movq xmm_dst, rax (zeros the high qword)
    t.push(0x5A); // pop rdx
    t.push(0x59); // pop rcx
    t.push(0x58); // pop rax
    t.push(0x9D); // popfq
    t
}

// =====================================================================================
// Windows-only glue: process state, executable memory, the patch-once bookkeeping.
// =====================================================================================

#[cfg(windows)]
use super::decode::ConsoleOnly;

#[cfg(windows)]
static SITES: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
/// Sites that could not be patched: emulated every time, not tried again.
#[cfg(windows)]
static NOPATCH: Mutex<Option<HashSet<usize>>> = Mutex::new(None);
#[cfg(windows)]
static STATE: Mutex<Arena> = Mutex::new(Arena { next: 0, end: 0, pads: Vec::new() });
#[cfg(windows)]
pub(crate) static PATCHED: AtomicU64 = AtomicU64::new(0);
#[cfg(windows)]
pub(crate) static REFUSED: AtomicU64 = AtomicU64::new(0);

#[cfg(windows)]
struct Arena {
    next: usize,
    end: usize,
    /// Padding runs taken (start, end).
    pads: Vec<(usize, usize)>,
}

#[cfg(windows)]
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_INSN_PATCH").map_or(true, |v| v != "0"))
}

/// A trap at `rip` (an illegal instruction there, or a breakpoint just after it) at a site
/// patched or being patched: the new `rip` to run again from.
#[cfg(windows)]
pub(crate) fn retry_at(rip: usize, breakpoint: bool) -> Option<usize> {
    let sites = SITES.lock().ok()?;
    let sites = sites.as_ref()?;
    if sites.contains(&rip) {
        return Some(rip);
    }
    if breakpoint && sites.contains(&(rip.wrapping_sub(1))) {
        return Some(rip - 1);
    }
    None
}

#[cfg(windows)]
fn write(at: usize, bytes: &[u8]) {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut old = 0u32;
    // SAFETY: code bytes of a site we decoded, made writable for the write and restored.
    unsafe {
        if VirtualProtect(at as *const _, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) != 0 {
            for (i, &b) in bytes.iter().enumerate() {
                std::ptr::write_volatile((at + i) as *mut u8, b);
            }
            VirtualProtect(at as *const _, bytes.len(), old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), at as *const _, bytes.len());
        }
    }
}

/// Replaces the `new.len()` bytes at `site`: first byte `int3`, then the rest, then the first.
#[cfg(windows)]
fn replace(site: usize, new: &[u8]) {
    write(site, &[0xCC]);
    if new.len() > 1 {
        write(site + 1, &new[1..]);
    }
    write(site, &new[..1]);
}

/// Room for `n` bytes of trampoline within ±2 GiB of `near`.
#[cfg(windows)]
fn trampoline(arena: &mut Arena, near: usize, n: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc};
    let fits = |at: usize| (at as i64 - near as i64).abs() < (1 << 31) - (1 << 20);
    if arena.next == 0 || arena.next + n > arena.end || !fits(arena.next) {
        const SIZE: usize = 1 << 20;
        let base = near & !0xFFFF;
        let mut got = 0usize;
        for k in 1..2000usize {
            for hint in [base.wrapping_sub(k * SIZE), base + k * SIZE] {
                // SAFETY: a fresh allocation at a free address or none.
                let p = unsafe { VirtualAlloc(hint as *const _, SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE) } as usize;
                if p != 0 && fits(p) {
                    got = p;
                    break;
                }
            }
            if got != 0 {
                break;
            }
        }
        if got == 0 {
            return None;
        }
        arena.next = got;
        arena.end = got + SIZE;
    }
    let at = arena.next;
    arena.next += (n + 15) & !15;
    Some(at)
}

/// A run of ≥ 5 `int3` within a short jump of `from_end` that is function padding (after a
/// `ret` or a jump, or longer than any immediate), not yet taken.
#[cfg(windows)]
fn padding(arena: &Arena, from_end: usize) -> Option<usize> {
    let lo = from_end.saturating_sub(128);
    let hi = from_end + 127;
    // The window's bytes: only within the site's committed code region.
    if !super::readable(lo, hi - lo + 1) {
        return None;
    }
    // SAFETY: checked readable.
    let w = unsafe { std::slice::from_raw_parts(lo as *const u8, hi - lo + 1) };
    let mut i = 0;
    while i < w.len() {
        if w[i] != 0xCC {
            i += 1;
            continue;
        }
        let s = i;
        while i < w.len() && w[i] == 0xCC {
            i += 1;
        }
        let run = i - s;
        let after_ret = s >= 1 && w[s - 1] == 0xC3;
        let after_jmp = (s >= 5 && w[s - 5] == 0xE9) || (s >= 2 && w[s - 2] == 0xEB);
        if run >= 5 && (after_ret || after_jmp || run >= 9) && s > 0 && i < w.len() {
            // the last five bytes of the run, if a short jump reaches them
            let at = lo + i - 5;
            let d = at as i64 - from_end as i64;
            let free = arena.pads.iter().all(|&(a, b)| at + 5 <= a || at >= b);
            if (-128..=127).contains(&d) && free {
                return Some(at);
            }
        }
    }
    None
}

/// Patches the site just emulated, so it does not trap again. `len`: the bytes of the
/// instruction to replace (for a clzero loop, the clzero alone).
#[cfg(windows)]
pub(crate) fn patch(site: usize, what: ConsoleOnly, len: usize, code: &[u8]) {
    if !enabled() {
        return;
    }
    if NOPATCH.lock().ok().is_some_and(|n| n.as_ref().is_some_and(|n| n.contains(&site))) {
        return;
    }
    {
        let mut sites = SITES.lock().unwrap_or_else(|e| e.into_inner());
        let sites = sites.get_or_insert_with(HashSet::new);
        if !sites.insert(site) {
            return;
        }
    }
    let ok = match what {
        ConsoleOnly::Clwb | ConsoleOnly::Monitorx => {
            replace(site, &fill_nop(len));
            true
        }
        ConsoleOnly::Mwaitx => {
            replace(site, &fill_pause(len));
            true
        }
        ConsoleOnly::Rdpru => {
            replace(site, &fill_rdtsc(len));
            true
        }
        ConsoleOnly::ClzeroLoop { .. } => {
            // the whole 12-byte loop → `jmp` to the same loop with the line zeroed by stores
            // (jumps reach it only at its head: the loop's own `jb`, or code before it)
            let i = code.iter().position(|&b| b == 0x0F).unwrap_or(0);
            let cmp = &code[i + 7..i + 10];
            let t = clzero_loop_body(cmp);
            via_jump(site, len, t)
        }
        ConsoleOnly::Clzero => {
            let a32 = code[..len - 3].contains(&0x67);
            let t = clzero_body(a32);
            via_padding(site, len, t)
        }
        ConsoleOnly::Rdpid(r) => {
            // the processor number: 0 (any number is a valid answer)
            let t = rdpid_body(r);
            via_padding(site, len, t)
        }
        // SSE4a immediate forms: bit field done by the trampoline; register forms (len/idx
        // taken from an XMM register) stay trapping for now.
        ConsoleOnly::Extrq { dst, src: None, len: ilen, idx } => {
            let t = extrq_body(dst, ilen, idx);
            via_padding(site, len, t)
        }
        ConsoleOnly::Insertq { dst, src, reg: false, len: ilen, idx } => {
            let t = insertq_body(dst, src, ilen, idx);
            via_padding(site, len, t)
        }
        _ => false,
    };
    if ok {
        let n = PATCHED.fetch_add(1, Relaxed) + 1;
        if n <= 64 || n.is_power_of_two() {
            tracing::info!(patched = n, instruction = ?what, at = format_args!("{site:#x}"), "a console CPU instruction patched (no trap there any more)");
        }
    } else {
        REFUSED.fetch_add(1, Relaxed);
        // stays trapping: emulated every time, as before
        let mut sites = SITES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = sites.as_mut() {
            s.remove(&site);
        }
        drop(sites);
        NOPATCH.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashSet::new).insert(site);
        static WARNED: AtomicU64 = AtomicU64::new(0);
        if WARNED.fetch_add(1, Relaxed) < 16 {
            tracing::warn!(instruction = ?what, at = format_args!("{site:#x}"), "a console CPU instruction left trapping (no room to patch it)");
        }
    }
}

/// `site` → `jmp short pad`; pad → `jmp trampoline`; trampoline = `body`; `jmp site+len`.
#[cfg(windows)]
fn via_padding(site: usize, len: usize, body: Vec<u8>) -> bool {
    let mut arena = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pad) = padding(&arena, site + 2) else { return false };
    let Some(tramp) = trampoline(&mut arena, site, body.len() + 5) else { return false };
    let Some(t) = trampoline_image(&body, tramp, site + len) else { return false };
    if rel32(pad + 5, tramp).is_none() {
        return false;
    }
    arena.pads.push((pad, pad + 5));
    drop(arena);
    // SAFETY: the trampoline is our own fresh executable memory.
    unsafe { std::ptr::copy_nonoverlapping(t.as_ptr(), tramp as *mut u8, t.len()) };
    write(pad, &pad_jump(pad, tramp)); // padding: never run before the site jumps there
    replace(site, &jmp_short_site(len, site, pad));
    true
}

/// `site` (`len` ≥ 5 bytes) → `jmp trampoline`; trampoline = `body`; `jmp site+len`.
#[cfg(windows)]
fn via_jump(site: usize, len: usize, body: Vec<u8>) -> bool {
    if len < 5 {
        return false;
    }
    let mut arena = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(tramp) = trampoline(&mut arena, site, body.len() + 5) else { return false };
    drop(arena);
    let Some(t) = trampoline_image(&body, tramp, site + len) else { return false };
    // SAFETY: the trampoline is our own fresh executable memory.
    unsafe { std::ptr::copy_nonoverlapping(t.as_ptr(), tramp as *mut u8, t.len()) };
    replace(site, &jmp_rel_site(len, site, tramp));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, OpKind, Register};

    /// 32-bit GPR variants: the rdpid trampoline loads a 32-bit immediate (zero-extended).
    const GPR32: [Register; 16] = [
        Register::EAX,
        Register::ECX,
        Register::EDX,
        Register::EBX,
        Register::ESP,
        Register::EBP,
        Register::ESI,
        Register::EDI,
        Register::R8D,
        Register::R9D,
        Register::R10D,
        Register::R11D,
        Register::R12D,
        Register::R13D,
        Register::R14D,
        Register::R15D,
    ];
    const XMM: [Register; 16] = [
        Register::XMM0,
        Register::XMM1,
        Register::XMM2,
        Register::XMM3,
        Register::XMM4,
        Register::XMM5,
        Register::XMM6,
        Register::XMM7,
        Register::XMM8,
        Register::XMM9,
        Register::XMM10,
        Register::XMM11,
        Register::XMM12,
        Register::XMM13,
        Register::XMM14,
        Register::XMM15,
    ];

    fn decode_all(bytes: &[u8]) -> Vec<iced_x86::Instruction> {
        let mut decoder = Decoder::with_ip(64, bytes, 0, DecoderOptions::NONE);
        let ins: Vec<_> = decoder.iter().collect();
        let consumed: usize = ins.iter().map(|i| i.len() as usize).sum();
        assert_eq!(consumed, bytes.len(), "undecoded bytes: {bytes:02x?}");
        ins
    }

    #[test]
    fn fills_are_byte_exact() {
        assert_eq!(fill_nop(3), vec![0x90, 0x90, 0x90]);
        assert_eq!(fill_pause(4), vec![0xF3, 0x90, 0x90, 0x90]);
        assert_eq!(fill_rdtsc(4), vec![0x0F, 0x31, 0x90, 0x90]);
    }

    #[test]
    fn rdpid_body_all_registers() {
        assert_eq!(rdpid_body(0), vec![0xB8, 0, 0, 0, 0]);
        assert_eq!(rdpid_body(7), vec![0xBF, 0, 0, 0, 0]);
        assert_eq!(rdpid_body(8), vec![0x41, 0xB8, 0, 0, 0, 0]);
        assert_eq!(rdpid_body(15), vec![0x41, 0xBF, 0, 0, 0, 0]);
        for r in 0..16u8 {
            let ins = decode_all(&rdpid_body(r));
            assert_eq!(ins.len(), 1, "r{r}");
            assert_eq!(ins[0].mnemonic(), Mnemonic::Mov, "r{r}: {}", ins[0]);
            assert_eq!(ins[0].op_register(0), GPR32[r as usize], "r{r}: {}", ins[0]);
            assert_eq!(ins[0].op_kind(1), OpKind::Immediate32, "r{r}: {}", ins[0]);
            assert_eq!(ins[0].immediate(1), 0, "r{r}: {}", ins[0]);
        }
    }

    #[test]
    fn rel32_range_boundaries() {
        // Exactly i32::MIN / i32::MAX fit; one past either end does not.
        let base = 0x1_0000_0000usize;
        assert_eq!(rel32(base, base + (i32::MAX as usize)), Some(i32::MAX.to_le_bytes()));
        assert_eq!(rel32(base, base - (i32::MIN.unsigned_abs() as usize)), Some(i32::MIN.to_le_bytes()));
        assert_eq!(rel32(base, base + (i32::MAX as usize) + 1), None);
        assert_eq!(rel32(base, base - (i32::MIN.unsigned_abs() as usize) - 1), None);
    }

    #[test]
    fn clzero_body_is_byte_exact() {
        // 64-bit form (no 0x67 prefix).
        assert_eq!(
            clzero_body(false),
            vec![
                0x9C, 0x51, 0x48, 0x89, 0xC1, 0x48, 0x83, 0xE1, 0xC0,
                0x48, 0xC7, 0x01, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x08, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x10, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x18, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x20, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x28, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x30, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x38, 0, 0, 0, 0,
                0x59, 0x9D,
            ]
        );
        // 32-bit form (0x67 address-size prefix at the site): mov ecx, eax = 89 C1.
        assert_eq!(&clzero_body(true)[2..4], &[0x89, 0xC1]);
        assert_eq!(clzero_body(true).len(), clzero_body(false).len() - 1);
    }

    #[test]
    fn clzero_body_decodes_both_address_sizes() {
        for a32 in [false, true] {
            let ins = decode_all(&clzero_body(a32));
            let m: Vec<_> = ins.iter().map(|i| i.mnemonic()).collect();
            assert_eq!(m.first(), Some(&Mnemonic::Pushfq), "{a32}: {m:?}");
            assert_eq!(m.last(), Some(&Mnemonic::Popfq), "{a32}: {m:?}");
            let mov = ins
                .iter()
                .find(|i| i.mnemonic() == Mnemonic::Mov && (i.op_register(0) == Register::ECX || i.op_register(0) == Register::RCX))
                .unwrap();
            if a32 {
                assert_eq!(mov.op_register(0), Register::ECX, "{mov}");
                assert_eq!(mov.op_register(1), Register::EAX, "{mov}");
            } else {
                assert_eq!(mov.op_register(0), Register::RCX, "{mov}");
                assert_eq!(mov.op_register(1), Register::RAX, "{mov}");
            }
        }
    }

    #[test]
    fn clzero_loop_body_is_byte_exact_and_decodes() {
        let body = clzero_loop_body(&[0x4C, 0x39, 0xD0]);
        assert_eq!(
            body,
            vec![
                0x51, 0x48, 0x89, 0xC1, 0x48, 0x83, 0xE1, 0xC0,
                0x48, 0xC7, 0x01, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x08, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x10, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x18, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x20, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x28, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x30, 0, 0, 0, 0,
                0x48, 0xC7, 0x41, 0x38, 0, 0, 0, 0,
                0x59, 0x48, 0x83, 0xC0, 0x40,
                0x4C, 0x39, 0xD0,
                0x72, 0xAF,
            ]
        );
        let ins = decode_all(&body);
        let m: Vec<_> = ins.iter().map(|i| i.mnemonic()).collect();
        assert_eq!(m[0], Mnemonic::Push);
        assert_eq!(m[1], Mnemonic::Mov);
        assert_eq!(m[2], Mnemonic::And);
        assert_eq!(m.iter().filter(|&&x| x == Mnemonic::Mov).count(), 9);
        assert_eq!(*m.last().unwrap(), Mnemonic::Jb);
        let cmp = ins.iter().find(|i| i.mnemonic() == Mnemonic::Cmp).unwrap();
        assert_eq!(cmp.op_register(0), Register::RAX);
        assert_eq!(cmp.op_register(1), Register::R10);
    }

    #[test]
    fn extrq_body_decodes_every_destination() {
        for dst in 0..16u8 {
            let body = extrq_body(dst, 8, 16);
            let ins = decode_all(&body);
            let m: Vec<_> = ins.iter().map(|i| i.mnemonic()).collect();
            assert_eq!(m[0], Mnemonic::Pushfq, "dst{dst}: {ins:?}");
            assert_eq!(*m.last().unwrap(), Mnemonic::Popfq, "dst{dst}: {ins:?}");
            let from = ins.iter().find(|i| i.mnemonic() == Mnemonic::Movq && i.op_register(1) == XMM[dst as usize]);
            assert!(from.is_some(), "dst{dst}: {ins:?}");
            let to = ins.iter().find(|i| i.mnemonic() == Mnemonic::Movq && i.op_register(0) == XMM[dst as usize]);
            assert!(to.is_some(), "dst{dst}: {ins:?}");
        }
        // len = 0 means 64: no mask `and`.
        let body = extrq_body(0, 0, 0);
        let ins = decode_all(&body);
        assert!(ins.iter().all(|i| i.mnemonic() != Mnemonic::And));
    }

    #[test]
    fn insertq_body_decodes_every_register_pair() {
        for dst in 0..16u8 {
            for src in 0..16u8 {
                let body = insertq_body(dst, src, 8, 16);
                let ins = decode_all(&body);
                let m: Vec<_> = ins.iter().map(|i| i.mnemonic()).collect();
                assert_eq!(m[0], Mnemonic::Pushfq, "dst{dst} src{src}: {ins:?}");
                assert_eq!(*m.last().unwrap(), Mnemonic::Popfq, "dst{dst} src{src}: {ins:?}");
                let read_src = ins.iter().find(|i| i.mnemonic() == Mnemonic::Movq && i.op_register(1) == XMM[src as usize]);
                assert!(read_src.is_some(), "dst{dst} src{src}: {ins:?}");
                let to = ins.iter().find(|i| i.mnemonic() == Mnemonic::Movq && i.op_register(0) == XMM[dst as usize]);
                assert!(to.is_some(), "dst{dst} src{src}: {ins:?}");
            }
        }
    }

    #[test]
    fn site_and_pad_jumps_decode() {
        let site = 0x1000usize;
        let tramp = 0x2000usize;
        let ins = decode_all(&jmp_rel_site(6, site, tramp));
        assert_eq!(ins[0].mnemonic(), Mnemonic::Jmp);
        assert_eq!(ins[0].op_kind(0), OpKind::NearBranch64);
        assert_eq!(jmp_rel_site(6, site, tramp), vec![0xE9, 0xFB, 0x0F, 0x00, 0x00, 0x90]);

        let pad = 0x1080usize;
        let ins = decode_all(&jmp_short_site(5, site, pad));
        assert_eq!(ins[0].mnemonic(), Mnemonic::Jmp);
        assert_eq!(ins[0].op_kind(0), OpKind::NearBranch64);
        assert_eq!(jmp_short_site(5, site, pad), vec![0xEB, 0x7E, 0x90, 0x90, 0x90]);

        let ins = decode_all(&pad_jump(pad, tramp));
        assert_eq!(ins[0].mnemonic(), Mnemonic::Jmp);
        assert_eq!(ins[0].op_kind(0), OpKind::NearBranch64);
        assert_eq!(pad_jump(pad, tramp), vec![0xE9, 0x7B, 0x0F, 0x00, 0x00]);
    }

    #[test]
    fn trampoline_image_appends_jmp_back() {
        let body = vec![0x9C, 0x9D];
        let img = trampoline_image(&body, 0x1000, 0x1030).unwrap();
        assert_eq!(&img[..2], &body[..]);
        assert_eq!(img[2], 0xE9);
        // rel32 from 0x1000 + 2 + 5 = 0x1007 to 0x1030 = 0x29.
        assert_eq!(&img[3..], &0x29u32.to_le_bytes());
        // Out of rel32 range: None (2^32 ahead does not fit in i32).
        assert!(trampoline_image(&body, 0, 0x1_0000_0000).is_none());
    }
}
