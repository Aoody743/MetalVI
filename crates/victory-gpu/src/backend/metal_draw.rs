//! P2: the title's draws on Metal (`VICTORY_METAL_DRAW=1`, with
//! `VICTORY_METAL=compute`).
//!
//! For now the draws are recorded on Metal *as well as* on the D3D12 path
//! (which still shows the picture): what Metal draws goes into its own
//! copies of the title's render targets, and is read back now and then
//! (`VICTORY_METAL_CAPTURE_EVERY`) so it can be compared with the D3D12
//! picture. Presenting from Metal comes once the two agree.
//!
//! - Pipelines: each title graphics pipeline's stages through Metal Shader
//!   Converter (disk-cached), the fixed-function state from F7's map, made
//!   on first use and kept per pipeline (as compute pipelines are).
//! - Recording (`draw.rs` `metal_draw`): everything resolved at record
//!   time, as for dispatches: per stage its top-level argument buffer, its
//!   one-view tables and the heap slots it reads; the targets as Metal
//!   textures; viewport, scissor, blend factor, stencil reference; the
//!   index buffer as a window of title memory (zero copy).
//! - Replay (`metal.rs` `run`): in list order with the dispatches;
//!   consecutive draws to the same targets share one render pass. Targets
//!   load and store (D3D12 keeps their contents); a clear is a pass of its
//!   own that clears.

use std::sync::Arc;

use victory_metal::Buffer;
use victory_metal::ir::{DescriptorEntry, TopLevel};
use victory_metal::render::{DepthStencilDesc, DepthStencilState, RenderPipeline, StencilFace};
use victory_metal::sys;
use victory_metal::texture::Texture;

use super::metal::{MetalBackend, convert_cached, render_desc, root_params, static_samplers};
use super::metal_render_map as map;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// Whether draws are recorded on Metal too.
pub(crate) fn draws_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::metal::mode() == super::metal::Mode::Compute && std::env::var("VICTORY_METAL_DRAW").is_ok_and(|v| v == "1")
    }) && !DRAWS_BROKEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// Set when a submission with draws faulted on the GPU: Metal then ignores
/// the process's submissions for a while, compute too, so draws stop.
pub(crate) static DRAWS_BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A title graphics pipeline made on Metal.
pub(crate) struct MetalGraphics {
    /// The title pipeline's shaders, for reports.
    pub(crate) name: String,
    pub(crate) pso: RenderPipeline,
    pub(crate) dss: DepthStencilState,
    pub(crate) depth_write: bool,
    /// Fill, cull, depth clip, winding (Metal values).
    pub(crate) raster: [u8; 4],
    /// Depth bias, slope scale, clamp.
    pub(crate) bias: [f32; 3],
    /// Per stage (in the host pipeline's order): its empty top level and
    /// where it binds (0 vertex, 1 fragment, 2 mesh).
    pub(crate) stages: Vec<(TopLevel, u8)>,
    /// Resident vertex-buffer slots required by the synthesized stage-in.
    pub(crate) vertex_slots: u32,
    pub(crate) mesh: bool,
    pub(crate) mesh_threads: [u64; 3],
    /// Non-none when compiled as a triangle-list geometry-emulation pipeline (for RECTANGLES).
    pub(crate) geometry: Option<victory_metal::msc::GeometryConfig>,
    /// What its targets must be (Metal pixel formats; 0 none) and their
    /// sample count.
    pub(crate) colors: [u32; 8],
    pub(crate) depth: u32,
    pub(crate) stencil: u32,
    pub(crate) samples: u8,
}

pub(crate) const STAGE_VERTEX: u8 = 0;
pub(crate) const STAGE_FRAGMENT: u8 = 1;
pub(crate) const STAGE_MESH: u8 = 2;
pub(crate) const STAGE_OBJECT: u8 = 3;

/// Dual-source blend factors (Metal 15..=18) as their first-source kind: a
/// pixel shader that does not write the second output (MSC gives it none)
/// cannot be made with them.
fn single_source(f: u8) -> u8 {
    match f {
        15 => 2,
        16 => 3,
        17 => 4,
        18 => 5,
        f => f,
    }
}

impl MetalBackend {
    /// The Metal pipeline of title graphics pipeline `key` (normal or rectangle-emulation), made on first use.
    pub(crate) fn graphics_pipeline(&self, key: usize, p: &HostPipeline, rects: bool) -> Option<Arc<MetalGraphics>> {
        if let Some(known) = self.graphics.lock().unwrap_or_else(|e| e.into_inner()).get(&(key, rects)) {
            return known.clone();
        }
        let made = self.make_graphics(p, rects);
        if made.is_err() {
            dump_failed(p);
        }
        if let Err(e) = &made {
            static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                tracing::warn!(pipeline = %p.name, rects, error = %e, "no Metal pipeline for a graphics pipeline");
            }
        }
        let made = made.ok().map(Arc::new);
        self.graphics.lock().unwrap_or_else(|e| e.into_inner()).insert((key, rects), made.clone());
        made
    }

    fn vertex_inputs<'a>(
        input: &'a map::PipelineStateInput,
        p: &HostPipeline,
        vertex_slots: &mut u32,
    ) -> Result<Vec<victory_metal::msc::VertexInput<'a>>, String> {
        use victory_metal::msc::{VERTEX_TABLE_SLOTS, VertexInput};
        if input.elements.is_empty() {
            return Ok(Vec::new());
        }
        if input.elements.len() != p.inputs.len() {
            return Err("Metal vertex layout disagrees with resolved host layout".into());
        }
        let mut elements = Vec::with_capacity(input.elements.len());
        for (source, resolved) in input.elements.iter().zip(&p.inputs) {
            if source.input_slot as usize >= VERTEX_TABLE_SLOTS
                || !matches!(source.slot_class, 0 | 1)
                || crate::resource::element(resolved.format).is_none()
            {
                return Err("unsupported Metal vertex input".into());
            }
            *vertex_slots |= 1u32 << source.input_slot;
            elements.push(VertexInput {
                semantic: &source.semantic_name,
                semantic_index: source.semantic_index,
                format: resolved.format,
                slot: source.input_slot,
                offset: resolved.offset,
                instance_step_rate: resolved.per_instance.then_some(source.instance_step_rate),
            });
        }
        Ok(elements)
    }

    fn make_graphics(&self, p: &HostPipeline, rects: bool) -> Result<MetalGraphics, String> {
        use victory_metal::msc::{Stage, convert_geometry, convert_geometry_vertex, convert_vertex_with_stage_in};
        let input = p.metal_input.as_ref().ok_or("no graphics state kept")?;
        let m = map::map_pipeline_state(input);
        map::check_integer_blending(&m.color)?;
        let _pool = victory_metal::Pool::new();
        let mut stages = Vec::new();
        let mut vertex = None;
        let mut fragment = None;
        let mut mesh = None;
        let mut stage_in = None;
        let mut geometry_config = None;
        let mut vertex_slots = 0;
        let mut mesh_threads = [1u64; 3];
        if rects {
            if p.mesh {
                return Err("mesh pipeline cannot draw rectangles".into());
            }
            let vs_stage = p.stages.iter().find(|s| s.shader.stage == "vertex").ok_or("no vertex stage for rectangles")?;
            let ps_stage = p.stages.iter().find(|s| s.shader.stage == "pixel");
            let vs_host = &vs_stage.translated().host;
            let ps_host = ps_stage.map(|s| &s.translated().host);
            let gs_dxil = victory_shader::emit::rect_list_gs(&vs_host.dxil, ps_host.map(|h| &h.dxil[..]))
                .map_err(|e| format!("rect_list_gs: {e}"))?;
            let elements = Self::vertex_inputs(input, p, &mut vertex_slots)?;
            let v = convert_geometry_vertex(&vs_host.dxil, &root_params(vs_host), &static_samplers(vs_host), &elements)
                .map_err(|e| e.to_string())?;
            let g = convert_geometry(&gs_dxil, &[], &[], v.vertex_output_size).map_err(|e| e.to_string())?;
            let olib = self.device.library(&v.shader.shader.metallib).map_err(|e| e.to_string())?;
            let sin = self.device.library(&v.shader.stage_in).map_err(|e| e.to_string())?;
            let mlib = self.device.library(&g.shader.metallib).map_err(|e| e.to_string())?;
            let f = ps_host.map(|h| convert_cached(h, Stage::Fragment)).transpose()?;
            let flib = f.as_ref().map(|c| self.device.library(&c.metallib).map_err(|e| e.to_string())).transpose()?;
            for s in &p.stages {
                match s.shader.stage {
                    "vertex" => stages.push((v.shader.shader.top_level.clone(), STAGE_OBJECT)),
                    "pixel" => {
                        if let Some(c) = &f {
                            stages.push((c.top_level.clone(), STAGE_FRAGMENT));
                        }
                    }
                    other => return Err(format!("{other} stage")),
                }
            }
            vertex = Some((olib, v.shader.shader.entry));
            mesh = Some((mlib, g.shader.entry));
            stage_in = Some(sin);
            if let (Some(lib), Some(c)) = (flib, f) {
                fragment = Some((lib, c.entry));
            }
            geometry_config = Some(g.config);
        } else {
            for s in &p.stages {
                let (stage, bind) = match s.shader.stage {
                    "vertex" => (Stage::Vertex, STAGE_VERTEX),
                    "pixel" => (Stage::Fragment, STAGE_FRAGMENT),
                    "mesh" => (Stage::Mesh, STAGE_MESH),
                    other => return Err(format!("{other} stage")),
                };
                let host = &s.translated().host;
                let c = if stage == Stage::Vertex && !input.elements.is_empty() {
                    let elements = Self::vertex_inputs(input, p, &mut vertex_slots)?;
                    // Kept with this graphics pipeline's layout, never in the
                    // cache for ordinary vertex shaders.
                    let compiled = convert_vertex_with_stage_in(&host.dxil, &root_params(host), &static_samplers(host), &elements)
                        .map_err(|e| e.to_string())?;
                    stage_in = Some(self.device.library(&compiled.stage_in).map_err(|e| e.to_string())?);
                    compiled.shader
                } else {
                    convert_cached(host, stage)?
                };
                let lib = self.device.library(&c.metallib).map_err(|e| e.to_string())?;
                stages.push((c.top_level.clone(), bind));
                match bind {
                    STAGE_VERTEX => vertex = Some((lib, c.entry)),
                    STAGE_FRAGMENT => fragment = Some((lib, c.entry)),
                    _ => {
                        mesh_threads = c.threadgroup.map(|t| u64::from(t.max(1)));
                        mesh = Some((lib, c.entry));
                    }
                }
            }
        }
        let mut desc = render_desc(&m);
        // `VICTORY_METAL_DEBUG_FRAGMENT=<text>` (diagnosis): pipelines whose
        // name contains <text> draw solid magenta instead of their pixel
        // shader, telling a draw that rasterises nothing from one whose
        // pixel shader writes zeros.
        if let Ok(want) = std::env::var("VICTORY_METAL_DEBUG_FRAGMENT")
            && !want.is_empty()
            && p.name.contains(&want)
            && fragment.is_some()
        {
            const DEBUG_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\nfragment float4 debug_fs() { return float4(1.0, 0.0, 1.0, 1.0); }\n";
            match self.device.library_msl(DEBUG_MSL) {
                Ok(lib) => {
                    tracing::warn!(pipeline = %p.name, "Metal: debug fragment (solid magenta) in place of the pixel shader");
                    // Its bindings stay: extra fragment buffers are harmless.
                    fragment = Some((lib, "debug_fs".to_string()));
                }
                Err(e) => tracing::warn!(error = %e, "Metal: the debug fragment did not compile"),
            }
        }
        let frag = fragment.as_ref().map(|(l, n)| (l, n.as_str()));
        let make = |desc: &victory_metal::render::RenderPipelineDesc| match (geometry_config, &vertex, &mesh, &stage_in) {
            (Some(cfg), Some((vl, vn)), Some((ml, mn)), Some(sin)) => {
                self.device.geometry_render_pipeline_desc((vl, vn), (ml, mn), sin, frag, desc, cfg)
            }
            (None, Some((l, n)), _, Some(stage_in)) => self.device.render_pipeline_stage_in((l, n), frag, desc, stage_in),
            (None, Some((l, n)), _, None) => self.device.render_pipeline_desc((l, n), frag, desc),
            (None, None, Some((l, n)), None) => self.device.mesh_render_pipeline_desc((l, n), frag, desc),
            _ => Err(victory_metal::Error("no vertex or mesh stage".into())),
        };
        let pso = match make(&desc) {
            Ok(pso) => pso,
            Err(e) if e.to_string().contains("index(1)") => {
                for c in &mut desc.colors {
                    c.src_rgb = single_source(c.src_rgb);
                    c.dst_rgb = single_source(c.dst_rgb);
                    c.src_alpha = single_source(c.src_alpha);
                    c.dst_alpha = single_source(c.dst_alpha);
                }
                make(&desc).map_err(|e| e.to_string())?
            }
            Err(e) => return Err(e.to_string()),
        };
        let face = |f: &map::StencilMap| StencilFace {
            compare: f.compare as u64 as u8,
            stencil_fail: f.stencil_fail as u64 as u8,
            depth_fail: f.depth_fail as u64 as u8,
            pass: f.depth_pass as u64 as u8,
            read_mask: f.read_mask,
            write_mask: f.write_mask,
        };
        let ds = &m.depth_stencil;
        let dss = self
            .depth_stencil(&DepthStencilDesc {
                depth_compare: ds.depth_compare as u64 as u8,
                depth_write: ds.depth_write,
                stencil: input.depth_stencil.stencil_enable,
                front: face(&ds.front),
                back: face(&ds.back),
            })
            .ok_or("no depth-stencil state")?;
        let r = &m.raster;
        Ok(MetalGraphics {
            name: p.name.clone(),
            pso,
            dss,
            depth_write: ds.depth_write,
            vertex_slots,
            raster: [
                r.fill.map_or(0, |v| v as u64 as u8),
                r.cull.map_or(0, |v| v as u64 as u8),
                r.depth_clip_mode as u64 as u8,
                r.winding as u64 as u8,
            ],
            bias: [r.depth_bias.depth_bias as f32, r.depth_bias.slope_scale, r.depth_bias.clamp],
            stages,
            mesh: mesh.is_some() && geometry_config.is_none(),
            mesh_threads,
            geometry: geometry_config,
            colors: desc.colors.map(|c| c.pixel_format),
            depth: desc.depth_format,
            stencil: desc.stencil_format,
            samples: desc.sample_count.max(1),
        })
    }

    /// A depth-stencil state, one per description.
    fn depth_stencil(&self, desc: &DepthStencilDesc) -> Option<DepthStencilState> {
        let mut known = self.depth_stencils.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = known.get(desc) {
            return Some(s.clone());
        }
        let s = self.device.depth_stencil_state(desc).ok()?;
        known.insert(*desc, s.clone());
        Some(s)
    }

    /// The attachment for a render target or depth view of title texture
    /// `resource` in DXGI format `format` at `level`/`slice`: its Metal
    /// texture, or a view of it in that format when Metal allows one.
    pub(crate) fn attachment(&self, resource: This, format: u32, level: u32, slice: u32, depth: bool) -> Option<Attachment> {
        let t = self.texture_for(resource)?;
        if level >= t.mips || slice >= t.slices.max(1) {
            return None;
        }
        let pixel = super::metal_texture::view_format(&t, format);
        if depth && !super::metal_texture::is_depth_pixel(t.pixel.0)
            && let Some(companion) = t.depth_companion(&self.device, format)
        {
            return Some(Attachment {
                resource: resource as usize,
                texture: companion.texture.clone(),
                pixel: companion.texture.pixel_format().0,
                level: level as u16,
                slice: slice as u16,
                width: (companion.texture.width() as u32 >> level).max(1),
                height: (companion.texture.height() as u32 >> level).max(1),
                samples: 1,
                companion: Some(companion),
            });
        }
        // A depth view needs a depth format and a target view a colour one
        // (the other way round is a GPU fault, not an error).
        if super::metal_texture::is_depth_pixel(pixel.0) != depth {
            static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                tracing::warn!(pixel = pixel.0, depth, format, "Metal: an attachment of the wrong kind of format, left out");
            }
            return None;
        }
        let texture = if pixel == t.pixel {
            t.texture.clone()
        } else {
            let key = (resource as usize, pixel.0);
            let mut views = self.target_views.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(v) = views.get(&key) {
                v.clone()
            } else {
                let _pool = victory_metal::Pool::new();
                let mut id = 0u64;
                // SAFETY: a live texture; every level and slice, its own type.
                let view = unsafe {
                    sys::MTLTexture_newTextureView(
                        t.texture.handle(),
                        pixel,
                        super::metal_texture::base_type(&t),
                        0,
                        t.mips as u16,
                        0,
                        t.slices.max(1) as u16,
                        super::metal_texture::identity(),
                        &mut id,
                    )
                };
                let view = Texture::from_object(victory_metal::Object::adopt(view)?);
                views.insert(key, view.clone());
                view
            }
        };
        let width = (t.desc.width as u32 >> level).max(1);
        let height = (t.desc.height.max(1) >> level).max(1);
        Some(Attachment { resource: resource as usize, texture, pixel: pixel.0, level: level as u16, slice: slice as u16, width, height, samples: t.desc.sample_count.max(1) as u8, companion: None })
    }
}

/// One render target or depth attachment.
#[derive(Clone)]
pub(crate) struct Attachment {
    /// The title resource it is a view of (for captures).
    pub(crate) resource: usize,
    pub(crate) texture: Texture,
    /// Its Metal pixel format.
    pub(crate) pixel: u32,
    pub(crate) level: u16,
    pub(crate) slice: u16,
    /// Size of that level.
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: u8,
    pub(crate) companion: Option<super::metal_depth::DepthCompanion>,
}

impl Attachment {
    fn same(&self, o: &Attachment) -> bool {
        self.texture.handle() == o.texture.handle() && self.level == o.level && self.slice == o.slice
    }
}

/// The targets of a run of draws.
#[derive(Clone, Default)]
pub(crate) struct Targets {
    pub(crate) colors: [Option<Attachment>; 8],
    pub(crate) depth: Option<Attachment>,
    /// The depth texture has a stencil plane (it is also the stencil
    /// attachment).
    pub(crate) stencil: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: u8,
}

impl Targets {
    /// Whether draws to `self` and `o` can share a render pass.
    pub(crate) fn same(&self, o: &Targets) -> bool {
        let eq = |a: &Option<Attachment>, b: &Option<Attachment>| match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => a.same(b),
            _ => false,
        };
        self.colors.iter().zip(&o.colors).all(|(a, b)| eq(a, b)) && eq(&self.depth, &o.depth)
    }

    /// Whether pipeline `g` can draw into these targets (formats and sample
    /// counts as Metal requires them; a mismatch is a GPU fault, not an
    /// error).
    pub(crate) fn fit(&self, g: &MetalGraphics) -> bool {
        // Exactly what the pipeline was made for, as D3D12 also requires: a
        // pass attachment the pipeline has no format for (or the other way
        // round) is a Metal validation error, undefined without the layer.
        let colors = self.colors.iter().zip(&g.colors).all(|(t, &f)| t.as_ref().map_or(0, |t| t.pixel) == f);
        let depth = self.depth.as_ref().map_or(0, |d| d.pixel) == g.depth;
        let stencil = if self.stencil { self.depth.as_ref().map_or(0, |d| d.pixel) } else { 0 } == g.stencil;
        colors && depth && stencil && self.samples.max(1) == g.samples.max(1)
    }

    /// Fills in the size and sample count from the attachments (the
    /// smallest size: D3D12 draws into the intersection).
    pub(crate) fn finish(&mut self) {
        let all = self.colors.iter().flatten().chain(self.depth.iter());
        let (mut w, mut h, mut s) = (u32::MAX, u32::MAX, 1);
        for a in all {
            w = w.min(a.width);
            h = h.min(a.height);
            s = a.samples;
        }
        if w == u32::MAX {
            (w, h) = (0, 0);
        }
        (self.width, self.height, self.samples) = (w, h, s);
    }
}

/// One stage's bindings for a draw (as a dispatch's).
pub(crate) struct StageBind {
    /// 0 vertex, 1 fragment, 2 mesh.
    pub(crate) bind: u8,
    pub(crate) top: Vec<u8>,
    pub(crate) tables: Vec<(u32, DescriptorEntry)>,
    pub(crate) heap: Option<Buffer>,
    pub(crate) sampler_heap: Option<Buffer>,
}

/// A draw as the title asks for it (before the index buffer is found).
pub(crate) enum DrawKindIn {
    Vertices { count: u32, instances: u32, first: u32, first_instance: u32 },
    Indexed { count: u32, instances: u32, first: u32, base: i32, first_instance: u32 },
    Mesh { groups: u32 },
    /// ExecuteIndirect draws: `max` records `stride` bytes apart at title
    /// address `records`, the count at `count` if given (F42).
    Indirect { indexed: bool, records: u64, stride: u32, max: u32, count: Option<u64> },
}

pub(crate) enum DrawKind {
    Vertices { count: u32, instances: u32, first: u32, first_instance: u32 },
    Indexed { count: u32, instances: u32, base: i32, first_instance: u32, buffer: Buffer, offset: u32, index_type: u8 },
    Mesh { groups: [u64; 3] },
    Geometry { draw: victory_metal::msc::GeometryDraw, _index: Option<Buffer> },
    /// A console topology Metal lacks (fan, quads, quad strip, line loop),
    /// drawn from 32-bit indices made at record time (relative to `base`,
    /// as a non-indexed draw's vertex ids are relative to its first vertex).
    Generated { indices: Vec<u32>, instances: u32, base: i32, first_instance: u32 },
    /// ExecuteIndirect: records translated on the GPU at replay (F42), one
    /// Metal indirect draw per record; `index`: the index buffer window,
    /// its type (0 16-bit, 1 32-bit) and capacity in indices.
    Indirect {
        records: Buffer,
        records_offset: u64,
        stride: u32,
        max: u32,
        count: Option<(Buffer, u64)>,
        index: Option<(Buffer, u64, u8, u32)>,
    },
}

/// One draw as recorded into a title list.
pub(crate) struct MetalDraw {
    pub(crate) pso: Arc<MetalGraphics>,
    pub(crate) targets: Arc<Targets>,
    pub(crate) stages: Vec<StageBind>,
    pub(crate) vertex: Option<Box<VertexFetch>>,
    pub(crate) viewport: sys::WMTViewport,
    pub(crate) scissor: sys::WMTScissorRect,
    pub(crate) blend: [f32; 4],
    pub(crate) stencil_ref: u8,
    /// MTLPrimitiveType.
    pub(crate) primitive: u8,
    pub(crate) kind: DrawKind,
}

pub(crate) struct VertexFetch {
    pub(crate) table: [u8; victory_metal::msc::VERTEX_TABLE_SLOTS * 16],
    /// Keep the addresses in `table` alive until replay completes.
    pub(crate) _buffers: Vec<Buffer>,
}

/// One clear as recorded into a title list.
pub(crate) struct MetalClear {
    pub(crate) target: Attachment,
    /// A color clear, or depth (and stencil when given) of a depth target.
    pub(crate) color: Option<[f64; 4]>,
    pub(crate) depth: Option<f32>,
    pub(crate) stencil: Option<u8>,
    /// The depth texture has a stencil plane.
    pub(crate) has_stencil: bool,
    /// Rectangles to clear (F49); empty: the whole view.
    pub(crate) rects: Vec<victory_metal::clear_resolve::ClearRect>,
}

/// Why draws did not go to Metal, counted for the log.
#[derive(Clone, Copy, Debug)]
pub(crate) enum DrawDrop {
    Ran,
    NoPipeline,
    Topology,
    Targets,
    Bind,
    IndexBuffer,
    Scissor,
    VertexFetch,
}

static DRAW_COUNTS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

pub(crate) fn count_draw(d: DrawDrop) {
    let n = DRAW_COUNTS[d as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let total: u64 = DRAW_COUNTS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).sum();
    if total.is_power_of_two() && total >= 64 {
        let c: Vec<u64> = DRAW_COUNTS.iter().map(|c| c.load(std::sync::atomic::Ordering::Relaxed)).collect();
        tracing::info!(
            draws = total,
            ran = c[0],
            no_pipeline = c[1],
            topology = c[2],
            targets = c[3],
            bind = c[4],
            index_buffer = c[5],
            scissor = c[6],
            vertex_fetch = c[7],
            "Metal draws: outcomes"
        );
    }
    let _ = n;
}

/// D3D12 render target / depth view words → (level, slice) of the view.
/// RTV dimensions: 2 1D, 3 1D array, 4 2D, 5 2D array, 6 2DMS, 7 2DMS array,
/// 8 3D; DSV (flags before the fields): 1 1D, 2 1D array, 3 2D, 4 2D array,
/// 5 2DMS, 6 2DMS array.
pub(crate) fn level_slice(words: &[u32; 6], depth: bool) -> Option<(u32, u32)> {
    let dim = words[1];
    let (level, slice) = if depth {
        // D3D12_DEPTH_STENCIL_VIEW_DESC: format, dimension, flags, then the
        // dimension's fields.
        match dim {
            1 | 3 => (words[3], 0),
            2 | 4 => (words[3], words[4]),
            5 => (0, 0),
            6 => (0, words[3]),
            _ => return None,
        }
    } else {
        match dim {
            2 | 4 => (words[2], 0),
            3 | 5 => (words[2], words[3]),
            6 => (0, 0),
            7 => (0, words[2]),
            _ => return None, // 3D targets: by depth plane, not yet
        }
    };
    Some((level, slice))
}

/// A D3D12 viewport as Metal's.
pub(crate) fn viewport(v: &windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT) -> sys::WMTViewport {
    sys::WMTViewport {
        originX: f64::from(v.TopLeftX),
        originY: f64::from(v.TopLeftY),
        width: f64::from(v.Width),
        height: f64::from(v.Height),
        znear: f64::from(v.MinDepth),
        zfar: f64::from(v.MaxDepth),
    }
}

/// A D3D12 scissor clamped into a `width`×`height` target (Metal requires
/// it inside); `None` if nothing is left.
pub(crate) fn scissor(r: &windows::Win32::Foundation::RECT, width: u32, height: u32) -> Option<sys::WMTScissorRect> {
    let l = r.left.clamp(0, width as i32) as u64;
    let t = r.top.clamp(0, height as i32) as u64;
    let rr = r.right.clamp(0, width as i32) as u64;
    let b = r.bottom.clamp(0, height as i32) as u64;
    (rr > l && b > t).then(|| sys::WMTScissorRect { x: l, y: t, width: rr - l, height: b - t })
}

/// The bytes MSC's vertex stages read at the draw-arguments bind point
/// (`IRRuntimeDrawParams`, 20 bytes) and the uniforms bind point (the index
/// type: 0 not indexed, 1 16-bit, 2 32-bit).
pub(crate) fn draw_params(kind: &DrawKind) -> ([u32; 5], u16) {
    match *kind {
        DrawKind::Vertices { count, instances, first, first_instance } => ([count, instances, first, first_instance, 0], 0),
        DrawKind::Indexed { count, instances, base, first_instance, offset, index_type, .. } => {
            // The direct Metal IR helper uses the buffer byte offset, not the D3D index number.
            ([count, instances, offset, base as u32, first_instance], u16::from(index_type) + 1)
        }
        DrawKind::Mesh { .. } | DrawKind::Geometry { .. } | DrawKind::Indirect { .. } => ([0; 5], 0),
        // the index offset is the ring's, filled in by `run`
        DrawKind::Generated { ref indices, instances, base, first_instance } => {
            ([indices.len() as u32, instances, 0, base as u32, first_instance], 2)
        }
    }
}

/// Every Metal render target this many presents is read back to
/// `VICTORY_METAL_CAPTURE_DIR` (`VICTORY_METAL_CAPTURE_EVERY`, 0: never).
pub(crate) fn capture_every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_METAL_CAPTURE_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

impl MetalBackend {
    /// Reads Metal texture `resource` back (level 0, slice 0) and writes it
    /// as a binary PPM (8-bit RGBA/BGRA formats only; others are skipped)
    /// to `<dir>/<name>.ppm`.
    pub(crate) fn capture(&self, resource: This, name: &str) {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};
        let Some(dir) = std::env::var_os("VICTORY_METAL_CAPTURE_DIR") else { return };
        let Some(t) = self.texture_for(resource) else { return };
        // Bytes per pixel of the formats written (others are skipped):
        // RGBA8 70/71, BGRA8 80/81, RGB10A2 90, RG11B10F 92, RGBA16F 115,
        // RGBA32F 125.
        let bpp: u32 = match t.pixel.0 {
            70 | 71 | 80 | 81 | 90 | 92 => 4,
            115 => 8,
            125 => 16,
            _ => return,
        };
        let format = t.pixel.0;
        let (w, h) = (t.desc.width as u32, t.desc.height.max(1));
        let row = (w * bpp).next_multiple_of(256);
        let bytes = (row * h) as usize;
        let _pool = victory_metal::Pool::new();
        // SAFETY: a fresh allocation, freed after the GPU is done.
        let mem = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if mem.is_null() {
            return;
        }
        // SAFETY: lives until the blit is waited for.
        if let Ok(buffer) = unsafe { self.device.wrap(mem, bytes as u64) }
            && let Ok(cb) = self.queue.command_buffer()
            && let Ok(mut blit) = cb.blit_encoder()
        {
            blit.copy_from_texture_to_buffer(&t.texture, 0, 0, [0, 0, 0], [u64::from(w), u64::from(h), 1], &buffer, 0, row, row * h);
            blit.finish();
            cb.commit();
            if cb.wait().is_ok() {
                // SAFETY: the GPU wrote `bytes` bytes there.
                let data = unsafe { std::slice::from_raw_parts(mem.cast::<u8>(), bytes) };
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let p = &data[y * row as usize + x * bpp as usize..][..bpp as usize];
                        out.extend_from_slice(&rgb8(format, p));
                    }
                }
                let path = std::path::Path::new(&dir).join(format!("{name}.ppm"));
                let _ = crate::make_dir(std::path::Path::new(&dir));
                if std::fs::write(&path, out).is_ok() {
                    tracing::info!(path = %path.display(), "Metal: a render target captured");
                }
            }
        }
        // SAFETY: ours, the GPU is done.
        unsafe { VirtualFree(mem, 0, MEM_RELEASE) };
    }
}

/// A buffer bound to the stage `bind` binds to.
pub(crate) fn stage_buffer(bind: u8, buffer: &Buffer, offset: u64, index: u8) -> victory_metal::render::Render<'_> {
    use victory_metal::render::Render;
    match bind {
        STAGE_VERTEX => Render::VertexBuffer { buffer, offset, index },
        STAGE_FRAGMENT => Render::FragmentBuffer { buffer, offset, index },
        STAGE_OBJECT => Render::ObjectBuffer { buffer, offset, index },
        _ => Render::MeshBuffer { buffer, offset, index },
    }
}

#[cfg(test)]
#[path = "metal_draw_helper_tests.rs"]
mod helper_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_words_give_level_and_slice() {
        // RTV 2D mip 3; 2D array mip 1 slice 5; 2DMS array slice 2; 3D: not yet.
        assert_eq!(level_slice(&[28, 4, 3, 0, 0, 0], false), Some((3, 0)));
        assert_eq!(level_slice(&[28, 5, 1, 5, 1, 0], false), Some((1, 5)));
        assert_eq!(level_slice(&[28, 7, 2, 1, 0, 0], false), Some((0, 2)));
        assert_eq!(level_slice(&[28, 8, 0, 0, 1, 0], false), None);
        // DSV (flags read-only depth 1, then the fields): 2D mip 2; 2D array
        // mip 0 slice 4; 2DMS; 2DMS array slice 3.
        assert_eq!(level_slice(&[40, 3, 1, 2, 0, 0], true), Some((2, 0)));
        assert_eq!(level_slice(&[40, 4, 0, 0, 4, 1], true), Some((0, 4)));
        assert_eq!(level_slice(&[40, 5, 3, 0, 0, 0], true), Some((0, 0)));
        assert_eq!(level_slice(&[40, 6, 0, 3, 1, 0], true), Some((0, 3)));
    }

    #[test]
    fn scissors_are_clamped_into_the_target() {
        use windows::Win32::Foundation::RECT;
        let s = scissor(&RECT { left: -10, top: 5, right: 5000, bottom: 300 }, 1920, 1080).unwrap();
        assert_eq!((s.x, s.y, s.width, s.height), (0, 5, 1920, 295));
        assert!(scissor(&RECT { left: 2000, top: 0, right: 2100, bottom: 10 }, 1920, 1080).is_none());
        assert!(scissor(&RECT { left: 10, top: 10, right: 10, bottom: 20 }, 1920, 1080).is_none());
    }

    #[test]
    fn draw_arguments_as_msc_reads_them() {
        let (a, u) = draw_params(&DrawKind::Vertices { count: 6, instances: 2, first: 3, first_instance: 1 });
        assert_eq!((a, u), ([6, 2, 3, 1, 0], 0));
        assert_eq!(draw_params(&DrawKind::Mesh { groups: [4, 1, 1] }).1, 0);
    }

    #[test]
    fn dual_source_factors_fall_back_to_the_first_source() {
        assert_eq!([15, 16, 17, 18, 1, 6].map(single_source), [2, 3, 4, 5, 1, 6]);
    }
}

/// The DXIL of a graphics pipeline Metal refused, kept for study in
/// `<VICTORY_METAL_CACHE>/refused/<pipeline>.<stage>.dxil` (once each).
fn dump_failed(p: &HostPipeline) {
    let Some(dir) = std::env::var_os("VICTORY_METAL_CACHE").map(std::path::PathBuf::from) else { return };
    let dir = dir.join("refused");
    if crate::make_dir(&dir).is_err() {
        return;
    }
    let stem: String = p.name.chars().filter(|c| c.is_ascii_alphanumeric()).take(80).collect();
    for s in &p.stages {
        let path = dir.join(format!("{stem}.{}.dxil", s.shader.stage));
        if crate::read_file(&path).is_err() {
            let _ = std::fs::write(&path, &s.translated().host.dxil);
        }
    }
}

/// Console topologies Metal has no primitive type for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConsoleTopology {
    TriangleFan,
    LineLoop,
    Quads,
    QuadStrip,
}

impl ConsoleTopology {
    /// MTLPrimitiveType of the lists they become (triangle 3, line 1).
    pub(crate) fn primitive(self) -> u8 {
        if self == ConsoleTopology::LineLoop { 1 } else { 3 }
    }
}

/// The list indices drawing `src` (the draw's vertex indices, in order) as
/// `t`: quads as (0,1,2)(0,2,3), quad strips as (0,1,3)(0,3,2) per step of
/// two, fans as (0,i,i+1), a line loop as its lines and the closing one.
/// Leftover vertices that make no whole primitive are dropped, as the
/// hardware does.
pub(crate) fn console_indices(t: ConsoleTopology, src: &[u32]) -> Vec<u32> {
    let n = src.len();
    let mut out = Vec::new();
    match t {
        ConsoleTopology::Quads => {
            for q in src.chunks_exact(4) {
                out.extend_from_slice(&[q[0], q[1], q[2], q[0], q[2], q[3]]);
            }
        }
        ConsoleTopology::QuadStrip => {
            let mut i = 0;
            while i + 3 < n {
                out.extend_from_slice(&[src[i], src[i + 1], src[i + 3], src[i], src[i + 3], src[i + 2]]);
                i += 2;
            }
        }
        ConsoleTopology::TriangleFan => {
            for i in 1..n.saturating_sub(1) {
                out.extend_from_slice(&[src[0], src[i], src[i + 1]]);
            }
        }
        ConsoleTopology::LineLoop => {
            if n >= 2 {
                for i in 0..n {
                    out.extend_from_slice(&[src[i], src[(i + 1) % n]]);
                }
            }
        }
    }
    out
}

/// Logs a target/pipeline mismatch once per kind (first 16 kinds).
pub(crate) fn note_mismatch(t: &Targets, g: &MetalGraphics) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let targets: Vec<u32> = t.colors.iter().map(|c| c.as_ref().map_or(0, |c| c.pixel)).collect();
    let key = format!(
        "targets {:?} depth {} stencil {} samples {} / pipeline {:?} depth {} stencil {} samples {}",
        targets,
        t.depth.as_ref().map_or(0, |d| d.pixel),
        t.stencil,
        t.samples,
        g.colors,
        g.depth,
        g.stencil,
        g.samples
    );
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < 16 && !seen.contains(&key) {
        tracing::info!(mismatch = %key, pipeline = %g.name, "Metal draws: targets do not fit the pipeline (first of its kind)");
        seen.push(key);
    }
}

#[cfg(test)]
mod console_topology_tests {
    use super::*;

    #[test]
    fn d3d_topology_numbers_are_the_real_ones() {
        use crate::backend::metal_map::{MtlPrimitiveType, map_topology};
        // d3dcommon.h: 1 point, 2 line list, 3 line strip, 4 triangle list,
        // 5 triangle strip, 10-13 adjacency (no Metal type).
        assert_eq!(map_topology(4).map(|t| t.primitive_type), Some(MtlPrimitiveType::Triangle));
        assert_eq!(map_topology(5).map(|t| t.primitive_type), Some(MtlPrimitiveType::TriangleStrip));
        assert_eq!(map_topology(3).map(|t| t.primitive_type), Some(MtlPrimitiveType::LineStrip));
        assert!(map_topology(12).is_none());
    }

    #[test]
    fn console_topologies_become_lists() {
        let v: Vec<u32> = (0..8).collect();
        assert_eq!(console_indices(ConsoleTopology::Quads, &v[..8]), [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
        assert_eq!(console_indices(ConsoleTopology::QuadStrip, &v[..6]), [0, 1, 3, 0, 3, 2, 2, 3, 5, 2, 5, 4]);
        assert_eq!(console_indices(ConsoleTopology::TriangleFan, &v[..5]), [0, 1, 2, 0, 2, 3, 0, 3, 4]);
        assert_eq!(console_indices(ConsoleTopology::LineLoop, &v[..3]), [0, 1, 1, 2, 2, 0]);
        // leftovers dropped; indexed sources map through
        assert_eq!(console_indices(ConsoleTopology::Quads, &[9, 8, 7, 6, 5]), [9, 8, 7, 9, 7, 6]);
        assert!(console_indices(ConsoleTopology::TriangleFan, &[1, 2]).is_empty());
    }
}

/// Logs `what` once per distinct text (first 32), for drop diagnostics.
pub(crate) fn note_once(what: String) {
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() < 32 && !seen.contains(&what) {
        tracing::info!(what = %what, "Metal draws: a drop reason (first of its kind)");
        seen.push(what);
    }
}

/// A texture-to-texture copy as recorded into a title list (F50's encoder
/// at replay): whole resources, or one subresource region.
pub(crate) struct MetalTexCopy {
    pub(crate) src: std::sync::Arc<super::metal_texture::MetalTexture>,
    pub(crate) dst: std::sync::Arc<super::metal_texture::MetalTexture>,
    pub(crate) whole: bool,
    pub(crate) src_slice: u32,
    pub(crate) src_level: u32,
    pub(crate) src_origin: [u64; 3],
    pub(crate) size: [u64; 3],
    pub(crate) dst_slice: u32,
    pub(crate) dst_level: u32,
    pub(crate) dst_origin: [u64; 3],
}

pub(crate) struct MetalResolve {
    pub(crate) src: Texture,
    pub(crate) src_slice: u16,
    pub(crate) dst: Texture,
    pub(crate) dst_level: u16,
    pub(crate) dst_slice: u16,
    pub(crate) format: sys::WMTPixelFormat,
    pub(crate) region: Option<victory_metal::clear_resolve::ResolveRegion>,
}

/// One pixel of a captured format as 8-bit RGB (floats clamped to 0..1).
fn rgb8(format: u32, p: &[u8]) -> [u8; 3] {
    let u = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let word = || u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
    match format {
        80 | 81 => [p[2], p[1], p[0]],
        90 => {
            let w = word();
            [((w & 1023) >> 2) as u8, (((w >> 10) & 1023) >> 2) as u8, (((w >> 20) & 1023) >> 2) as u8]
        }
        92 => {
            // R11 G11 B10 floats: 5-bit exponent, 6/6/5-bit mantissas
            let w = word();
            let f = |bits: u32, mant: u32| -> f32 {
                let e = (bits >> mant) & 31;
                let m = bits & ((1 << mant) - 1);
                if e == 0 {
                    m as f32 / (1 << mant) as f32 * 2f32.powi(-14)
                } else {
                    (1.0 + m as f32 / (1 << mant) as f32) * 2f32.powi(e as i32 - 15)
                }
            };
            [u(f(w & 2047, 6)), u(f((w >> 11) & 2047, 6)), u(f((w >> 22) & 1023, 5))]
        }
        115 => {
            let h = |i: usize| half_to_f32(u16::from_le_bytes([p[i], p[i + 1]]));
            [u(h(0)), u(h(2)), u(h(4))]
        }
        125 => {
            let f = |i: usize| f32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
            [u(f(0)), u(f(4)), u(f(8))]
        }
        _ => [p[0], p[1], p[2]],
    }
}

fn half_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((h >> 10) & 31);
    let m = f32::from(h & 1023);
    s * match e {
        0 => m / 1024.0 * 2f32.powi(-14),
        31 => f32::INFINITY,
        _ => (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The colour targets Metal drew into since the last capture: resource →
/// pixel area (for captures of intermediate targets).
pub(crate) static DRAWN_TARGETS: std::sync::Mutex<Vec<(usize, u64)>> = std::sync::Mutex::new(Vec::new());

/// Title textures handed to the window (`present`), for the final-draw
/// dossier below.
pub(crate) static PRESENTED: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Notes a presented texture.
pub(crate) fn note_presented(resource: usize) {
    let mut p = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
    if !p.contains(&resource) {
        p.push(resource);
    }
}

/// The final-draw dossier (when capturing): the presented texture a draw
/// recorded just now targets, if any; `run` captures it right after the
/// submission holding that draw finishes.
pub(crate) static CAPTURE_AFTER: std::sync::Mutex<Vec<(usize, String)>> = std::sync::Mutex::new(Vec::new());

/// Logs (the first times) and marks for a capture a draw into a presented
/// texture.
pub(crate) fn final_draw_dossier(d: &MetalDraw) {
    if capture_every() == 0 {
        return;
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let presented = PRESENTED.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let Some(target) = d.targets.colors.iter().flatten().find(|a| presented.contains(&a.resource)) else { return };
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n >= 6 && !n.is_power_of_two() {
        return;
    }
    let (args, index_type) = draw_params(&d.kind);
    let others: Vec<String> = d
        .targets
        .colors
        .iter()
        .flatten()
        .map(|a| format!("{}:{}x{} px{}", a.resource, a.width, a.height, a.pixel))
        .collect();
    let v = d.viewport;
    let s = d.scissor;
    tracing::info!(
        n,
        pipeline = %d.pso.name,
        colors = ?d.pso.colors,
        depth_format = d.pso.depth,
        depth_attached = d.targets.depth.is_some(),
        depth_write = d.pso.depth_write,
        raster = ?d.pso.raster,
        primitive = d.primitive,
        args = ?args,
        index_type,
        viewport = format_args!("{:.0},{:.0} {:.0}x{:.0} z{:.2}-{:.2}", v.originX, v.originY, v.width, v.height, v.znear, v.zfar),
        scissor = format_args!("{},{} {}x{}", s.x, s.y, s.width, s.height),
        targets = ?others,
        stages = d.stages.len(),
        tables = ?d.stages.iter().map(|b| (b.bind, b.tables.len(), b.heap.is_some())).collect::<Vec<_>>(),
        vertex_fetch = d.vertex.is_some(),
        "Metal final-draw dossier: a draw into the presented texture"
    );
    CAPTURE_AFTER.lock().unwrap_or_else(|e| e.into_inner()).push((target.resource, format!("final-draw{n:03}")));
}

/// What wrote each title texture since the last capture (only when
/// capturing): draws, clears, copies on Metal; resolves left on D3D12
/// and D3D12-only buffer copies.
pub(crate) static WRITES: std::sync::Mutex<Vec<(usize, [u32; 9])>> = std::sync::Mutex::new(Vec::new());

pub(crate) const WRITE_NAMES: [&str; 9] = ["draw", "clear", "copy", "resolve_d3d_only", "buffer_copy_d3d_only", "uav_view", "draw_asked", "draw_no_host_pipeline", "draw_not_ready"];

/// Notes a write of kind `kind` (an index into [`WRITE_NAMES`]) to `resource`.
pub(crate) fn note_write(resource: usize, kind: usize) {
    if capture_every() == 0 {
        return;
    }
    let mut w = WRITES.lock().unwrap_or_else(|e| e.into_inner());
    match w.iter_mut().find(|(r, _)| *r == resource) {
        Some((_, n)) => n[kind] += 1,
        None => {
            let mut n = [0; 9];
            n[kind] = 1;
            w.push((resource, n));
        }
    }
}

/// Notes the colour targets of a recorded draw (only when capturing).
pub(crate) fn note_targets(t: &Targets) {
    if capture_every() == 0 {
        return;
    }
    let mut d = DRAWN_TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    for a in t.colors.iter().flatten() {
        note_write(a.resource, 0);
        if !d.iter().any(|(r, _)| *r == a.resource) {
            d.push((a.resource, u64::from(a.width) * u64::from(a.height)));
        }
    }
}
