//! The native Metal backend (metal-backend/DESIGN.md), built beside the
//! D3D12 one. Pure translation tables and view parsing are portable; the
//! parts that touch Metal go through victory-metal (Windows/Wine only).

// Not wired into the frame path yet (P1-4): kept compiled and tested.
#[allow(dead_code)]
pub mod metal_map;
#[allow(dead_code)]
pub mod metal_views;
#[cfg(windows)]
#[allow(dead_code)]
pub mod metal_heap;
#[cfg(windows)]
#[allow(dead_code)]
pub mod metal;
#[cfg(windows)]
#[allow(dead_code)]
pub mod metal_texture;
#[cfg(windows)]
#[allow(dead_code)]
pub mod metal_detile;
#[cfg(windows)]
pub mod metal_draw;
#[cfg(windows)]
pub mod metal_present;
// P2: graphics pipeline state → Metal descriptions (F7).
#[allow(dead_code)]
pub mod metal_render_map;
