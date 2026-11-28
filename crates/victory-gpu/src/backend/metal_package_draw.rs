//! T3 F4: textures and the draw-side ops of a frame package.
//!
//! - Textures are named by the title resource they belong to (the key of
//!   `MetalBackend::textures`) and recorded with the Metal description they
//!   were made with; a [`Ref::Tex`] names the texture itself, a target view
//!   of it in another pixel format, or its depth companion.
//! - Contents: every texture a run references, before its first run (GPU
//!   writable ones: render target, depth, unordered access) or again
//!   whenever its bytes changed (title-filled ones, refilled by the CPU);
//!   the ones the frame may have written again at its end (the reference).
//!   Multisampled textures and stencil twins are not read (counted).
//! - Draw, Clear, Copy, Resolve, FloatRowReadback and Ambient are encoded
//!   here; graphics pipelines are rebuilt from their converted libraries and
//!   descriptions ([`GraphicsSource`]).

use std::collections::{BTreeSet, HashMap};
use std::io::Write;
use std::sync::{Arc, Mutex};

use victory_metal::Buffer;
use victory_metal::render::{ColorDesc, DepthStencilDesc, RenderPipelineDesc, StencilFace};
use victory_metal::sys::{self, WMTPixelFormat, WMTTextureType, WMTTextureUsage};
use victory_metal::texture::{Texture, TextureDescriptor};

use super::{put_blob, put_u32, put_u64, PTop, Reader, Ref, TEntry};
use crate::backend::metal::MetalBackend;
use crate::backend::metal_draw::{Attachment, DrawKind, MetalDraw, MetalGraphics};
use crate::backend::metal_texture::MetalTexture;
use crate::resource::ResourceDesc;

/// [`Ref::Tex`] views.
pub(crate) const VIEW_SELF: u64 = 0;
/// `VIEW_TARGET | pixel`: the texture viewed in another pixel format (every
/// level and slice), as `MetalBackend::attachment` makes for targets.
pub(crate) const VIEW_TARGET: u64 = 1 << 32;
pub(crate) const VIEW_COMPANION: u64 = 2 << 32;

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PAttachment {
    pub(crate) tex: Ref,
    pub(crate) pixel: u32,
    pub(crate) level: u16,
    pub(crate) slice: u16,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: u8,
    pub(crate) companion: bool,
    pub(crate) twin: bool,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub(crate) struct PTargets {
    pub(crate) colors: [Option<PAttachment>; 8],
    pub(crate) depth: Option<PAttachment>,
    pub(crate) stencil: bool,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: u8,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PStage {
    pub(crate) bind: u8,
    pub(crate) top: PTop,
    pub(crate) tables: Vec<(u32, TEntry)>,
    pub(crate) heap: Ref,
    pub(crate) sampler_heap: Ref,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PDrawKind {
    Vertices { count: u32, instances: u32, first: u32, first_instance: u32 },
    Indexed { count: u32, instances: u32, base: i32, first_instance: u32, buffer: Ref, index_type: u8 },
    Mesh { groups: [u64; 3] },
    Generated { indices: Vec<u32>, instances: u32, base: i32, first_instance: u32 },
    Indirect {
        records: Ref,
        records_va: u64,
        stride: u32,
        max: u32,
        count: Option<Ref>,
        count_va: Option<u64>,
        /// The index buffer window, its type and capacity in indices.
        index: Option<(Ref, u8, u32)>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PDraw {
    pub(crate) pipeline: u32,
    pub(crate) targets: PTargets,
    pub(crate) stages: Vec<PStage>,
    /// The vertex table: per slot (address, length, stride).
    pub(crate) vertex: Option<Vec<(Ref, u32, u32)>>,
    pub(crate) viewport: [f64; 6],
    pub(crate) scissor: [u64; 4],
    pub(crate) blend: [f32; 4],
    pub(crate) stencil_ref: u8,
    pub(crate) primitive: u8,
    pub(crate) kind: PDrawKind,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PClear {
    pub(crate) target: PAttachment,
    pub(crate) color: Option<[f64; 4]>,
    pub(crate) depth: Option<f32>,
    pub(crate) stencil: Option<u8>,
    pub(crate) has_stencil: bool,
    pub(crate) rects: Vec<[i32; 4]>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PCopy {
    pub(crate) src: Ref,
    pub(crate) dst: Ref,
    pub(crate) whole: bool,
    pub(crate) src_slice: u32,
    pub(crate) src_level: u32,
    pub(crate) src_origin: [u64; 3],
    pub(crate) size: [u64; 3],
    pub(crate) dst_slice: u32,
    pub(crate) dst_level: u32,
    pub(crate) dst_origin: [u64; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PResolve {
    pub(crate) src: Ref,
    pub(crate) src_slice: u16,
    pub(crate) dst: Ref,
    pub(crate) dst_level: u16,
    pub(crate) dst_slice: u16,
    pub(crate) format: u32,
    /// src origin, dst origin, size.
    pub(crate) region: Option<[u32; 6]>,
}

fn put_u8(w: &mut Vec<u8>, v: u8) {
    w.push(v);
}
fn put_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(w: &mut Vec<u8>, v: f32) {
    put_u32(w, v.to_bits());
}
fn put_f64(w: &mut Vec<u8>, v: f64) {
    put_u64(w, v.to_bits());
}
fn get_f32(r: &mut Reader<'_>) -> Option<f32> {
    r.u32().map(f32::from_bits)
}
fn get_f64(r: &mut Reader<'_>) -> Option<f64> {
    r.u64().map(f64::from_bits)
}

fn put_attachment(w: &mut Vec<u8>, a: &PAttachment) {
    a.tex.put(w);
    put_u32(w, a.pixel);
    put_u16(w, a.level);
    put_u16(w, a.slice);
    put_u32(w, a.width);
    put_u32(w, a.height);
    put_u8(w, a.samples);
    put_u8(w, a.companion as u8 | (a.twin as u8) << 1);
}

fn get_attachment(r: &mut Reader<'_>) -> Option<PAttachment> {
    let tex = Ref::get(r)?;
    let (pixel, level, slice, width, height, samples, flags) = (r.u32()?, r.u16()?, r.u16()?, r.u32()?, r.u32()?, r.u8()?, r.u8()?);
    Some(PAttachment { tex, pixel, level, slice, width, height, samples, companion: flags & 1 != 0, twin: flags & 2 != 0 })
}

fn put_opt_attachment(w: &mut Vec<u8>, a: &Option<PAttachment>) {
    put_u8(w, a.is_some() as u8);
    if let Some(a) = a {
        put_attachment(w, a);
    }
}

fn get_opt_attachment(r: &mut Reader<'_>) -> Option<Option<PAttachment>> {
    Some(if r.u8()? != 0 { Some(get_attachment(r)?) } else { None })
}

fn put_opt_ref(w: &mut Vec<u8>, v: &Option<Ref>) {
    put_u8(w, v.is_some() as u8);
    v.unwrap_or(Ref::Null).put(w);
}

fn get_opt_ref(r: &mut Reader<'_>) -> Option<Option<Ref>> {
    let some = r.u8()? != 0;
    let v = Ref::get(r)?;
    Some(some.then_some(v))
}

fn put_opt_u64(w: &mut Vec<u8>, v: Option<u64>) {
    put_u8(w, v.is_some() as u8);
    put_u64(w, v.unwrap_or(0));
}

fn get_opt_u64(r: &mut Reader<'_>) -> Option<Option<u64>> {
    let some = r.u8()? != 0;
    let v = r.u64()?;
    Some(some.then_some(v))
}

pub(crate) fn put_stage(w: &mut Vec<u8>, s: &PStage) {
    put_u8(w, s.bind);
    s.heap.put(w);
    s.sampler_heap.put(w);
    put_blob(w, &s.top.raw);
    put_u32(w, s.top.refs.len() as u32);
    for (at, r) in &s.top.refs {
        put_u32(w, *at);
        r.put(w);
    }
    put_u32(w, s.tables.len() as u32);
    for (at, e) in &s.tables {
        put_u32(w, *at);
        e.put(w);
    }
}

pub(crate) fn get_stage(r: &mut Reader<'_>) -> Option<PStage> {
    let bind = r.u8()?;
    let heap = Ref::get(r)?;
    let sampler_heap = Ref::get(r)?;
    let raw = r.blob()?.to_vec();
    let n = r.u32()?;
    let mut refs = Vec::new();
    for _ in 0..n {
        refs.push((r.u32()?, Ref::get(r)?));
    }
    let n = r.u32()?;
    let mut tables = Vec::new();
    for _ in 0..n {
        tables.push((r.u32()?, TEntry::get(r)?));
    }
    Some(PStage { bind, top: PTop { raw, refs }, tables, heap, sampler_heap })
}

pub(crate) fn put_draw(w: &mut Vec<u8>, d: &PDraw) {
    put_u32(w, d.pipeline);
    for c in &d.targets.colors {
        put_opt_attachment(w, c);
    }
    put_opt_attachment(w, &d.targets.depth);
    put_u8(w, d.targets.stencil as u8);
    put_u32(w, d.targets.width);
    put_u32(w, d.targets.height);
    put_u8(w, d.targets.samples);
    put_u32(w, d.stages.len() as u32);
    for s in &d.stages {
        put_stage(w, s);
    }
    put_u8(w, d.vertex.is_some() as u8);
    if let Some(v) = &d.vertex {
        put_u32(w, v.len() as u32);
        for (r, len, stride) in v {
            r.put(w);
            put_u32(w, *len);
            put_u32(w, *stride);
        }
    }
    for v in d.viewport {
        put_f64(w, v);
    }
    for v in d.scissor {
        put_u64(w, v);
    }
    for v in d.blend {
        put_f32(w, v);
    }
    put_u8(w, d.stencil_ref);
    put_u8(w, d.primitive);
    match &d.kind {
        PDrawKind::Vertices { count, instances, first, first_instance } => {
            put_u8(w, 0);
            for v in [*count, *instances, *first, *first_instance] {
                put_u32(w, v);
            }
        }
        PDrawKind::Indexed { count, instances, base, first_instance, buffer, index_type } => {
            put_u8(w, 1);
            for v in [*count, *instances, *base as u32, *first_instance] {
                put_u32(w, v);
            }
            buffer.put(w);
            put_u8(w, *index_type);
        }
        PDrawKind::Mesh { groups } => {
            put_u8(w, 2);
            for g in groups {
                put_u64(w, *g);
            }
        }
        PDrawKind::Generated { indices, instances, base, first_instance } => {
            put_u8(w, 3);
            put_u32(w, indices.len() as u32);
            for i in indices {
                put_u32(w, *i);
            }
            for v in [*instances, *base as u32, *first_instance] {
                put_u32(w, v);
            }
        }
        PDrawKind::Indirect { records, records_va, stride, max, count, count_va, index } => {
            put_u8(w, 4);
            records.put(w);
            put_u64(w, *records_va);
            put_u32(w, *stride);
            put_u32(w, *max);
            put_opt_ref(w, count);
            put_opt_u64(w, *count_va);
            put_u8(w, index.is_some() as u8);
            let (r, t, c) = index.unwrap_or((Ref::Null, 0, 0));
            r.put(w);
            put_u8(w, t);
            put_u32(w, c);
        }
    }
}

pub(crate) fn get_draw(r: &mut Reader<'_>) -> Option<PDraw> {
    let pipeline = r.u32()?;
    let mut targets = PTargets::default();
    for c in &mut targets.colors {
        *c = get_opt_attachment(r)?;
    }
    targets.depth = get_opt_attachment(r)?;
    targets.stencil = r.u8()? != 0;
    targets.width = r.u32()?;
    targets.height = r.u32()?;
    targets.samples = r.u8()?;
    let n = r.u32()?;
    let mut stages = Vec::new();
    for _ in 0..n {
        stages.push(get_stage(r)?);
    }
    let vertex = if r.u8()? != 0 {
        let n = r.u32()?;
        let mut v = Vec::new();
        for _ in 0..n {
            v.push((Ref::get(r)?, r.u32()?, r.u32()?));
        }
        Some(v)
    } else {
        None
    };
    let viewport = [get_f64(r)?, get_f64(r)?, get_f64(r)?, get_f64(r)?, get_f64(r)?, get_f64(r)?];
    let scissor = [r.u64()?, r.u64()?, r.u64()?, r.u64()?];
    let blend = [get_f32(r)?, get_f32(r)?, get_f32(r)?, get_f32(r)?];
    let stencil_ref = r.u8()?;
    let primitive = r.u8()?;
    let kind = match r.u8()? {
        0 => PDrawKind::Vertices { count: r.u32()?, instances: r.u32()?, first: r.u32()?, first_instance: r.u32()? },
        1 => {
            let (count, instances, base, first_instance) = (r.u32()?, r.u32()?, r.u32()? as i32, r.u32()?);
            PDrawKind::Indexed { count, instances, base, first_instance, buffer: Ref::get(r)?, index_type: r.u8()? }
        }
        2 => PDrawKind::Mesh { groups: [r.u64()?, r.u64()?, r.u64()?] },
        3 => {
            let n = r.u32()?;
            let mut indices = Vec::with_capacity(n.min(1 << 20) as usize);
            for _ in 0..n {
                indices.push(r.u32()?);
            }
            PDrawKind::Generated { indices, instances: r.u32()?, base: r.u32()? as i32, first_instance: r.u32()? }
        }
        4 => {
            let records = Ref::get(r)?;
            let records_va = r.u64()?;
            let stride = r.u32()?;
            let max = r.u32()?;
            let count = get_opt_ref(r)?;
            let count_va = get_opt_u64(r)?;
            let some = r.u8()? != 0;
            let index = (Ref::get(r)?, r.u8()?, r.u32()?);
            PDrawKind::Indirect { records, records_va, stride, max, count, count_va, index: some.then_some(index) }
        }
        _ => return None,
    };
    Some(PDraw { pipeline, targets, stages, vertex, viewport, scissor, blend, stencil_ref, primitive, kind })
}

pub(crate) fn put_clear(w: &mut Vec<u8>, c: &PClear) {
    put_attachment(w, &c.target);
    put_u8(w, c.color.is_some() as u8);
    for v in c.color.unwrap_or([0.0; 4]) {
        put_f64(w, v);
    }
    put_u8(w, c.depth.is_some() as u8);
    put_f32(w, c.depth.unwrap_or(0.0));
    put_u8(w, c.stencil.is_some() as u8);
    put_u8(w, c.stencil.unwrap_or(0));
    put_u8(w, c.has_stencil as u8);
    put_u32(w, c.rects.len() as u32);
    for r in &c.rects {
        for v in r {
            put_u32(w, *v as u32);
        }
    }
}

pub(crate) fn get_clear(r: &mut Reader<'_>) -> Option<PClear> {
    let target = get_attachment(r)?;
    let some = r.u8()? != 0;
    let color = [get_f64(r)?, get_f64(r)?, get_f64(r)?, get_f64(r)?];
    let dsome = r.u8()? != 0;
    let depth = get_f32(r)?;
    let ssome = r.u8()? != 0;
    let stencil = r.u8()?;
    let has_stencil = r.u8()? != 0;
    let n = r.u32()?;
    let mut rects = Vec::new();
    for _ in 0..n {
        rects.push([r.u32()? as i32, r.u32()? as i32, r.u32()? as i32, r.u32()? as i32]);
    }
    Some(PClear { target, color: some.then_some(color), depth: dsome.then_some(depth), stencil: ssome.then_some(stencil), has_stencil, rects })
}

pub(crate) fn put_copy(w: &mut Vec<u8>, c: &PCopy) {
    c.src.put(w);
    c.dst.put(w);
    put_u8(w, c.whole as u8);
    for v in [c.src_slice, c.src_level, c.dst_slice, c.dst_level] {
        put_u32(w, v);
    }
    for v in c.src_origin.iter().chain(&c.size).chain(&c.dst_origin) {
        put_u64(w, *v);
    }
}

pub(crate) fn get_copy(r: &mut Reader<'_>) -> Option<PCopy> {
    let (src, dst, whole) = (Ref::get(r)?, Ref::get(r)?, r.u8()? != 0);
    let (src_slice, src_level, dst_slice, dst_level) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
    let mut v = [0u64; 9];
    for x in &mut v {
        *x = r.u64()?;
    }
    Some(PCopy {
        src,
        dst,
        whole,
        src_slice,
        src_level,
        src_origin: [v[0], v[1], v[2]],
        size: [v[3], v[4], v[5]],
        dst_slice,
        dst_level,
        dst_origin: [v[6], v[7], v[8]],
    })
}

pub(crate) fn put_resolve(w: &mut Vec<u8>, c: &PResolve) {
    c.src.put(w);
    put_u16(w, c.src_slice);
    c.dst.put(w);
    put_u16(w, c.dst_level);
    put_u16(w, c.dst_slice);
    put_u32(w, c.format);
    put_u8(w, c.region.is_some() as u8);
    for v in c.region.unwrap_or([0; 6]) {
        put_u32(w, v);
    }
}

pub(crate) fn get_resolve(r: &mut Reader<'_>) -> Option<PResolve> {
    let src = Ref::get(r)?;
    let src_slice = r.u16()?;
    let dst = Ref::get(r)?;
    let (dst_level, dst_slice, format) = (r.u16()?, r.u16()?, r.u32()?);
    let some = r.u8()? != 0;
    let mut v = [0u32; 6];
    for x in &mut v {
        *x = r.u32()?;
    }
    Some(PResolve { src, src_slice, dst, dst_level, dst_slice, format, region: some.then_some(v) })
}

pub(crate) fn draw_refs(d: &PDraw, out: &mut Vec<Ref>) {
    for a in d.targets.colors.iter().flatten().chain(d.targets.depth.iter()) {
        out.push(a.tex);
    }
    for s in &d.stages {
        out.push(s.heap);
        out.push(s.sampler_heap);
        out.extend(s.top.refs.iter().map(|(_, r)| *r));
        for (_, e) in &s.tables {
            out.push(e.gpu);
            out.push(e.view);
        }
    }
    if let Some(v) = &d.vertex {
        out.extend(v.iter().map(|(r, _, _)| *r));
    }
    match &d.kind {
        PDrawKind::Indexed { buffer, .. } => out.push(*buffer),
        PDrawKind::Indirect { records, count, index, .. } => {
            out.push(*records);
            out.extend(*count);
            out.extend(index.map(|i| i.0));
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Graphics pipeline sources
// ---------------------------------------------------------------------------

/// A Metal library as made: metallib bytes, or Metal source (tests).
#[derive(Clone, Debug)]
pub(crate) enum Lib {
    Metallib(Vec<u8>),
    Msl(String),
}

/// What a graphics pipeline's Metal pipeline was made from.
#[derive(Clone, Debug)]
pub(crate) struct GraphicsSource {
    pub(crate) vertex: Option<(Lib, String)>,
    pub(crate) fragment: Option<(Lib, String)>,
    pub(crate) mesh: Option<(Lib, String)>,
    pub(crate) stage_in: Option<Lib>,
    pub(crate) desc: RenderPipelineDesc,
    pub(crate) dss: DepthStencilDesc,
    /// Per stage that has them: (bind, root parameters, static samplers),
    /// bound again on the rebuilt top levels (their table is this process's).
    pub(crate) static_samplers: Vec<(u8, u32, Vec<victory_metal::msc::StaticSampler>)>,
}

/// Sources by `MetalGraphics` (its `Arc`'s address), kept while recording is wanted.
static GSOURCES: Mutex<Option<HashMap<usize, Arc<GraphicsSource>>>> = Mutex::new(None);

pub(crate) fn register_graphics(g: &Arc<MetalGraphics>, source: GraphicsSource) {
    GSOURCES.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(Arc::as_ptr(g) as usize, Arc::new(source));
}

fn graphics_source(g: &Arc<MetalGraphics>) -> Option<Arc<GraphicsSource>> {
    GSOURCES.lock().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&(Arc::as_ptr(g) as usize)).cloned()
}

/// Hook (`MetalBackend::graphics_pipeline`): keeps what a graphics pipeline
/// was made from when recording is wanted, converted again the way
/// `make_graphics` converts it (its converter outputs are cached).
/// Rectangle emulation and the debug fragment are not kept: their draws
/// replay as pipelines without a source (skipped and counted).
pub(crate) fn note_graphics(made: &Option<Arc<MetalGraphics>>, p: &crate::pipeline::host::HostPipeline, rects: bool) {
    if super::settings().is_none() || rects {
        return;
    }
    let Some(g) = made else { return };
    match derive(p) {
        Ok(src) => register_graphics(g, src),
        Err(e) => tracing::debug!(pipeline = %p.name, error = %e, "frame package: no source kept for a graphics pipeline"),
    }
}

fn derive(p: &crate::pipeline::host::HostPipeline) -> Result<GraphicsSource, String> {
    use crate::backend::metal::{convert_cached, render_desc, root_params, static_samplers};
    use crate::backend::metal_render_map as map;
    use victory_metal::msc::{Stage, VERTEX_TABLE_SLOTS, VertexInput, convert_vertex_with_stage_in};
    let input = p.metal_input.as_ref().ok_or("no graphics state kept")?;
    let mm = map::map_pipeline_state(input);
    let (mut vertex, mut fragment, mut mesh, mut stage_in) = (None, None, None, None);
    let mut statics = Vec::new();
    for s in &p.stages {
        let stage = match s.shader.stage {
            "vertex" => Stage::Vertex,
            "pixel" => Stage::Fragment,
            "mesh" => Stage::Mesh,
            other => return Err(format!("{other} stage")),
        };
        let host = &s.translated().host;
        let c = if stage == Stage::Vertex && !input.elements.is_empty() {
            if input.elements.len() != p.inputs.len() {
                return Err("vertex layout".into());
            }
            let mut elements = Vec::new();
            for (source, resolved) in input.elements.iter().zip(&p.inputs) {
                if source.input_slot as usize >= VERTEX_TABLE_SLOTS || !matches!(source.slot_class, 0 | 1) {
                    return Err("vertex input".into());
                }
                elements.push(VertexInput {
                    semantic: &source.semantic_name,
                    semantic_index: source.semantic_index,
                    format: resolved.format,
                    slot: source.input_slot,
                    offset: resolved.offset,
                    instance_step_rate: resolved.per_instance.then_some(source.instance_step_rate),
                });
            }
            let compiled = convert_vertex_with_stage_in(&host.dxil, &root_params(host), &static_samplers(host), &elements).map_err(|e| e.to_string())?;
            stage_in = Some(Lib::Metallib(compiled.stage_in));
            compiled.shader
        } else {
            convert_cached(host, stage)?
        };
        let lib = (Lib::Metallib(c.metallib), c.entry);
        let samplers = static_samplers(host);
        if !samplers.is_empty() {
            let bind = match stage {
                Stage::Vertex => crate::backend::metal_draw::STAGE_VERTEX,
                Stage::Fragment => crate::backend::metal_draw::STAGE_FRAGMENT,
                _ => crate::backend::metal_draw::STAGE_MESH,
            };
            statics.push((bind, root_params(host).len() as u32, samplers));
        }
        match stage {
            Stage::Vertex => vertex = Some(lib),
            Stage::Fragment => fragment = Some(lib),
            _ => mesh = Some(lib),
        }
    }
    let face = |f: &map::StencilMap| StencilFace {
        compare: f.compare as u64 as u8,
        stencil_fail: f.stencil_fail as u64 as u8,
        depth_fail: f.depth_fail as u64 as u8,
        pass: f.depth_pass as u64 as u8,
        read_mask: f.read_mask,
        write_mask: f.write_mask,
    };
    let ds = &mm.depth_stencil;
    let dss = DepthStencilDesc {
        depth_compare: ds.depth_compare as u64 as u8,
        depth_write: ds.depth_write,
        stencil: input.depth_stencil.stencil_enable,
        front: face(&ds.front),
        back: face(&ds.back),
    };
    Ok(GraphicsSource { vertex, fragment, mesh, stage_in, desc: render_desc(&mm), dss, static_samplers: statics })
}

fn put_lib(w: &mut Vec<u8>, l: Option<&Lib>) {
    match l {
        None => put_u8(w, 0),
        Some(Lib::Metallib(b)) => {
            put_u8(w, 1);
            put_blob(w, b);
        }
        Some(Lib::Msl(s)) => {
            put_u8(w, 2);
            put_blob(w, s.as_bytes());
        }
    }
}

fn get_lib(r: &mut Reader<'_>) -> Option<Option<Lib>> {
    Some(match r.u8()? {
        0 => None,
        1 => Some(Lib::Metallib(r.blob()?.to_vec())),
        2 => Some(Lib::Msl(r.text()?)),
        _ => return None,
    })
}

fn put_fn(w: &mut Vec<u8>, f: &Option<(Lib, String)>) {
    put_lib(w, f.as_ref().map(|(l, _)| l));
    put_blob(w, f.as_ref().map_or("", |(_, n)| n.as_str()).as_bytes());
}

fn get_fn(r: &mut Reader<'_>) -> Option<Option<(Lib, String)>> {
    let lib = get_lib(r)?;
    let name = r.text()?;
    Some(lib.map(|l| (l, name)))
}

fn put_desc(w: &mut Vec<u8>, d: &RenderPipelineDesc) {
    for c in &d.colors {
        put_u32(w, c.pixel_format);
        w.extend_from_slice(&[c.blending as u8, c.src_rgb, c.dst_rgb, c.op_rgb, c.src_alpha, c.dst_alpha, c.op_alpha, c.write_mask]);
    }
    put_u32(w, d.depth_format);
    put_u32(w, d.stencil_format);
    put_u8(w, d.sample_count);
    put_u8(w, d.alpha_to_coverage as u8);
    put_u32(w, d.topology_class);
}

fn get_desc(r: &mut Reader<'_>) -> Option<RenderPipelineDesc> {
    let mut d = RenderPipelineDesc::default();
    for c in &mut d.colors {
        let pixel_format = r.u32()?;
        let b = r.bytes(8)?;
        *c = ColorDesc { pixel_format, blending: b[0] != 0, src_rgb: b[1], dst_rgb: b[2], op_rgb: b[3], src_alpha: b[4], dst_alpha: b[5], op_alpha: b[6], write_mask: b[7] };
    }
    d.depth_format = r.u32()?;
    d.stencil_format = r.u32()?;
    d.sample_count = r.u8()?;
    d.alpha_to_coverage = r.u8()? != 0;
    d.topology_class = r.u32()?;
    Some(d)
}

fn put_face(w: &mut Vec<u8>, f: &StencilFace) {
    w.extend_from_slice(&[f.compare, f.stencil_fail, f.depth_fail, f.pass, f.read_mask, f.write_mask]);
}

fn get_face(r: &mut Reader<'_>) -> Option<StencilFace> {
    let b = r.bytes(6)?;
    Some(StencilFace { compare: b[0], stencil_fail: b[1], depth_fail: b[2], pass: b[3], read_mask: b[4], write_mask: b[5] })
}

/// One graphics pipeline's record in `graphics.bin`.
pub(crate) fn put_graphics(w: &mut Vec<u8>, id: u32, g: &MetalGraphics, source: Option<&GraphicsSource>) {
    put_u32(w, id);
    put_blob(w, g.name.as_bytes());
    put_u8(w, g.depth_write as u8);
    w.extend_from_slice(&g.raster);
    for b in g.bias {
        put_f32(w, b);
    }
    put_u32(w, g.stages.len() as u32);
    for (top, bind) in &g.stages {
        put_u8(w, *bind);
        put_blob(w, &top.layout_bytes());
        put_blob(w, &top.bytes);
    }
    put_u32(w, g.vertex_slots);
    put_u8(w, g.mesh as u8);
    for t in g.mesh_threads {
        put_u64(w, t);
    }
    put_u8(w, g.geometry.is_some() as u8);
    for c in g.colors {
        put_u32(w, c);
    }
    put_u32(w, g.depth);
    put_u32(w, g.stencil);
    put_u8(w, g.samples);
    match source {
        None => put_u8(w, 0),
        Some(s) => {
            put_u8(w, 1);
            put_fn(w, &s.vertex);
            put_fn(w, &s.fragment);
            put_fn(w, &s.mesh);
            put_lib(w, s.stage_in.as_ref());
            put_desc(w, &s.desc);
            w.extend_from_slice(&[s.dss.depth_compare, s.dss.depth_write as u8, s.dss.stencil as u8]);
            put_face(w, &s.dss.front);
            put_face(w, &s.dss.back);
            put_u32(w, s.static_samplers.len() as u32);
            for (bind, params, samplers) in &s.static_samplers {
                put_u8(w, *bind);
                put_u32(w, *params);
                super::put_samplers(w, samplers);
            }
        }
    }
}

pub(crate) struct GraphicsDef {
    pub(crate) id: u32,
    pub(crate) name: String,
    pub(crate) depth_write: bool,
    pub(crate) raster: [u8; 4],
    pub(crate) bias: [f32; 3],
    pub(crate) stages: Vec<(Vec<u8>, Vec<u8>, u8)>,
    pub(crate) vertex_slots: u32,
    pub(crate) mesh: bool,
    pub(crate) mesh_threads: [u64; 3],
    pub(crate) geometry: bool,
    pub(crate) colors: [u32; 8],
    pub(crate) depth: u32,
    pub(crate) stencil: u32,
    pub(crate) samples: u8,
    pub(crate) source: Option<GraphicsSource>,
}

pub(crate) fn get_graphics(r: &mut Reader<'_>) -> Option<GraphicsDef> {
    let id = r.u32()?;
    let name = r.text()?;
    let depth_write = r.u8()? != 0;
    let raster: [u8; 4] = r.bytes(4)?.try_into().ok()?;
    let bias = [get_f32(r)?, get_f32(r)?, get_f32(r)?];
    let n = r.u32()?;
    let mut stages = Vec::new();
    for _ in 0..n {
        let bind = r.u8()?;
        stages.push((r.blob()?.to_vec(), r.blob()?.to_vec(), bind));
    }
    let vertex_slots = r.u32()?;
    let mesh = r.u8()? != 0;
    let mesh_threads = [r.u64()?, r.u64()?, r.u64()?];
    let geometry = r.u8()? != 0;
    let mut colors = [0u32; 8];
    for c in &mut colors {
        *c = r.u32()?;
    }
    let (depth, stencil, samples) = (r.u32()?, r.u32()?, r.u8()?);
    let source = if r.u8()? != 0 {
        let (vertex, fragment, mesh, stage_in) = (get_fn(r)?, get_fn(r)?, get_fn(r)?, get_lib(r)?);
        let desc = get_desc(r)?;
        let b = r.bytes(3)?;
        let (depth_compare, depth_write, stencil) = (b[0], b[1] != 0, b[2] != 0);
        let dss = DepthStencilDesc { depth_compare, depth_write, stencil, front: get_face(r)?, back: get_face(r)? };
        let n = r.u32()?;
        let mut static_samplers = Vec::new();
        for _ in 0..n {
            static_samplers.push((r.u8()?, r.u32()?, super::get_samplers(r)?));
        }
        Some(GraphicsSource { vertex, fragment, mesh, stage_in, desc, dss, static_samplers })
    } else {
        None
    };
    Some(GraphicsDef { id, name, depth_write, raster, bias, stages, vertex_slots, mesh, mesh_threads, geometry, colors, depth, stencil, samples, source })
}

/// Dual-source blend factors as their first-source kind (as `make_graphics`
/// retries with when Metal refuses the pipeline).
fn single_source(f: u8) -> u8 {
    match f {
        15 => 2,
        16 => 3,
        17 => 4,
        18 => 5,
        f => f,
    }
}

/// The graphics pipeline of `def`, made anew.
pub(crate) fn build_graphics(m: &MetalBackend, def: &GraphicsDef) -> Result<MetalGraphics, String> {
    use victory_metal::ir::TopLevel;
    let s = def.source.as_ref().ok_or("no source recorded")?;
    if def.geometry {
        return Err("rectangle emulation is not replayed".into());
    }
    let lib = |l: &Lib| match l {
        Lib::Metallib(b) => m.device.library(b).map_err(|e| e.to_string()),
        Lib::Msl(src) => m.device.library_msl(src).map_err(|e| e.to_string()),
    };
    let f = |x: &Option<(Lib, String)>| -> Result<Option<(victory_metal::Object, String)>, String> { x.as_ref().map(|(l, n)| Ok((lib(l)?, n.clone()))).transpose() };
    let (vertex, fragment, mesh) = (f(&s.vertex)?, f(&s.fragment)?, f(&s.mesh)?);
    let stage_in = s.stage_in.as_ref().map(lib).transpose()?;
    let frag = fragment.as_ref().map(|(l, n)| (l, n.as_str()));
    let make = |desc: &RenderPipelineDesc| match (&vertex, &mesh, &stage_in) {
        (Some((l, n)), _, Some(sin)) => m.device.render_pipeline_stage_in((l, n), frag, desc, sin),
        (Some((l, n)), _, None) => m.device.render_pipeline_desc((l, n), frag, desc),
        (None, Some((l, n)), None) => m.device.mesh_render_pipeline_desc((l, n), frag, desc),
        _ => Err(victory_metal::Error("no vertex or mesh stage".into())),
    };
    let pso = match make(&s.desc) {
        Ok(p) => p,
        Err(e) if e.to_string().contains("index(1)") => {
            let mut desc = s.desc;
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
    let dss = m.device.depth_stencil_state(&s.dss).map_err(|e| e.to_string())?;
    let mut stages = Vec::new();
    for (layout, template, bind) in &def.stages {
        let (mut top, _) = TopLevel::from_layout_bytes(layout).ok_or("bad stage layout")?;
        if top.bytes.len() == template.len() {
            top.bytes.clone_from(template);
        }
        if let Some((_, params, samplers)) = s.static_samplers.iter().find(|(b, _, _)| b == bind) {
            m.bind_static_samplers(&mut top, *params as usize, samplers)?;
        }
        stages.push((top, *bind));
    }
    Ok(MetalGraphics {
        cost_ns: Default::default(),
        name: def.name.clone(),
        pso,
        dss,
        depth_write: def.depth_write,
        raster: def.raster,
        bias: def.bias,
        stages,
        vertex_slots: def.vertex_slots,
        mesh: def.mesh,
        mesh_threads: def.mesh_threads,
        geometry: None,
        colors: def.colors,
        depth: def.depth,
        stencil: def.stencil,
        samples: def.samples,
    })
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/// A texture's record (`textures.defs`).
#[derive(Clone, Copy)]
pub(crate) struct TexDef {
    pub(crate) id: u32,
    /// The title resource (`MetalBackend::textures` key).
    pub(crate) key: u64,
    pub(crate) desc: ResourceDesc,
    pub(crate) pixel: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth: u32,
    pub(crate) mips: u32,
    pub(crate) slices: u32,
    pub(crate) type_: u32,
    pub(crate) samples: u32,
    pub(crate) usage: u32,
    pub(crate) twin: bool,
}

fn desc_bytes(d: &ResourceDesc) -> &[u8] {
    // SAFETY: an integer-only C record, read as bytes.
    unsafe { std::slice::from_raw_parts(std::ptr::from_ref(d).cast(), size_of::<ResourceDesc>()) }
}

fn desc_from(b: &[u8]) -> Option<ResourceDesc> {
    if b.len() != size_of::<ResourceDesc>() {
        return None;
    }
    // SAFETY: an integer-only C record; every bit pattern is valid.
    let mut d: ResourceDesc = unsafe { std::mem::zeroed() };
    // SAFETY: sizes checked above.
    unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), std::ptr::from_mut(&mut d).cast::<u8>(), b.len()) };
    Some(d)
}

const FLAG_RENDER_TARGET: u32 = 1;
const FLAG_DEPTH: u32 = 2;
const FLAG_UNORDERED: u32 = 4;

impl TexDef {
    /// The record of `t` (its Metal description as `make_texture` makes it).
    pub(crate) fn of(id: u32, key: usize, t: &MetalTexture) -> Self {
        let d = &t.desc;
        let mut usage = 1 | 0x10;
        if d.flags & FLAG_UNORDERED != 0 {
            usage |= 2;
        }
        if d.flags & (FLAG_RENDER_TARGET | FLAG_DEPTH) != 0 {
            usage |= 4;
        }
        Self {
            id,
            key: key as u64,
            desc: *d,
            pixel: t.pixel.0,
            width: t.texture.width() as u32,
            height: t.texture.height() as u32,
            depth: t.texture.depth() as u32,
            mips: t.mips,
            slices: t.slices,
            type_: crate::backend::metal_texture::base_type(t).0,
            samples: d.sample_count.max(1),
            usage,
            twin: t.twin.is_some(),
        }
    }

    /// Whether the GPU may write it (target, depth, unordered access).
    pub(crate) fn writable(&self) -> bool {
        self.desc.flags & (FLAG_RENDER_TARGET | FLAG_DEPTH | FLAG_UNORDERED) != 0
    }

    pub(crate) fn put(&self, w: &mut Vec<u8>) {
        put_u32(w, self.id);
        put_u64(w, self.key);
        put_blob(w, desc_bytes(&self.desc));
        for v in [self.pixel, self.width, self.height, self.depth, self.mips, self.slices, self.type_, self.samples, self.usage] {
            put_u32(w, v);
        }
        put_u8(w, self.twin as u8);
    }

    pub(crate) fn get(r: &mut Reader<'_>) -> Option<Self> {
        let id = r.u32()?;
        let key = r.u64()?;
        let desc = desc_from(r.blob()?)?;
        let mut v = [0u32; 9];
        for x in &mut v {
            *x = r.u32()?;
        }
        Some(Self {
            id,
            key,
            desc,
            pixel: v[0],
            width: v[1],
            height: v[2],
            depth: v[3],
            mips: v[4],
            slices: v[5],
            type_: v[6],
            samples: v[7],
            usage: v[8],
            twin: r.u8()? != 0,
        })
    }

    /// Its subresources as stored and the blob's size; `None` for what is
    /// not read (multisampled, formats without a known layout).
    pub(crate) fn layout(&self) -> Option<(Vec<Sub>, u64)> {
        if self.samples > 1 {
            return None;
        }
        let planes = planes(self.pixel)?;
        let is_3d = self.type_ == WMTTextureType::WMTTextureType3D.0;
        let mut out = Vec::new();
        let mut at = 0u64;
        for level in 0..self.mips {
            let w = (self.width >> level).max(1);
            let h = (self.height >> level).max(1);
            let d = if is_3d { (self.depth >> level).max(1) } else { 1 };
            for slice in 0..if is_3d { 1 } else { self.slices.max(1) } {
                for &(bytes, bw, bh, option) in &planes {
                    let row = w.div_ceil(bw) * bytes;
                    let image = row * h.div_ceil(bh);
                    out.push(Sub { level, slice, option, size: [u64::from(w), u64::from(h), u64::from(d)], row, image, offset: at });
                    at = (at + u64::from(image) * u64::from(d)).next_multiple_of(256);
                }
            }
        }
        Some((out, at))
    }
}

/// One subresource (and plane) of a stored texture.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sub {
    pub(crate) level: u32,
    pub(crate) slice: u32,
    pub(crate) option: u16,
    pub(crate) size: [u64; 3],
    pub(crate) row: u32,
    pub(crate) image: u32,
    pub(crate) offset: u64,
}

/// The planes a texture of Metal format `pixel` is read in: (bytes per
/// block, block width, height, blit option).
fn planes(pixel: u32) -> Option<Vec<(u32, u32, u32, u16)>> {
    use victory_metal::blit::{BLIT_OPTION_DEPTH, BLIT_OPTION_NONE, BLIT_OPTION_STENCIL};
    Some(match pixel {
        250 => vec![(2, 1, 1, BLIT_OPTION_NONE)],
        252 => vec![(4, 1, 1, BLIT_OPTION_NONE)],
        253 => vec![(1, 1, 1, BLIT_OPTION_NONE)],
        260 => vec![(4, 1, 1, BLIT_OPTION_DEPTH), (1, 1, 1, BLIT_OPTION_STENCIL)],
        _ => {
            use crate::backend::metal_map::{format_props, map_pixel_format};
            let dxgi = (0..=200).find(|&d| map_pixel_format(d).is_some_and(|f| f as u64 == u64::from(pixel)))?;
            let p = format_props(dxgi)?;
            if p.block_size_bytes == 0 || p.depth_stencil {
                return None;
            }
            vec![(p.block_size_bytes, p.block_width.max(1), p.block_height.max(1), BLIT_OPTION_NONE)]
        }
    })
}

/// The bytes of `texture` laid out as `def` says (one command buffer of
/// blits, waited for through the fuse-checked path).
pub(crate) fn read_texture(m: &MetalBackend, texture: &Texture, def: &TexDef) -> Option<Vec<u8>> {
    let (subs, total) = def.layout()?;
    let (buffer, mem) = m.device.shared_buffer_mapped(&vec![0u8; total.max(256) as usize]).ok()?;
    let _pool = victory_metal::Pool::new();
    let cb = m.queue.command_buffer().ok()?;
    let mut blit = cb.blit_encoder().ok()?;
    for s in &subs {
        blit.copy_from_texture_to_buffer_with_options(texture, s.slice, s.level as u16, [0; 3], s.size, &buffer, s.offset, s.row, s.image, s.option);
    }
    blit.finish();
    crate::backend::metal::submit_checked(&cb, &[], true, "frame package texture read").ok()?;
    // SAFETY: the shared buffer holds `total` bytes the GPU is done writing.
    Some(unsafe { std::slice::from_raw_parts(mem.cast_const(), total as usize) }.to_vec())
}

/// Writes `bytes` (laid out as `def` says) into `texture`.
pub(crate) fn write_texture(m: &MetalBackend, texture: &Texture, def: &TexDef, bytes: &[u8]) -> bool {
    let Some((subs, total)) = def.layout() else { return false };
    if bytes.len() as u64 != total {
        return false;
    }
    let Ok(buffer) = m.device.shared_buffer(bytes) else { return false };
    let _pool = victory_metal::Pool::new();
    let Ok(cb) = m.queue.command_buffer() else { return false };
    let Ok(mut blit) = cb.blit_encoder() else { return false };
    for s in &subs {
        blit.copy_from_buffer_to_texture_with_options(&buffer, s.offset, s.row, s.image, s.size, texture, s.slice, s.level as u16, [0; 3], s.option);
    }
    blit.finish();
    crate::backend::metal::submit_checked(&cb, &[], true, "frame package texture write").is_ok()
}

/// A texture of `def` made anew (private storage, resident), with stencil
/// twins when it had them.
pub(crate) fn make_texture(m: &MetalBackend, def: &TexDef) -> Result<MetalTexture, String> {
    let texture = m
        .device
        .new_texture(&TextureDescriptor {
            pixel_format: WMTPixelFormat(def.pixel),
            width: def.width,
            height: def.height,
            depth: def.depth.max(1),
            array_length: if def.type_ == WMTTextureType::WMTTextureType2DArray.0 { def.slices } else { 1 },
            type_: WMTTextureType(def.type_),
            mipmap_level_count: def.mips,
            sample_count: def.samples,
            usage: WMTTextureUsage(def.usage),
            options: sys::WMTResourceOptions(0x20),
        })
        .map_err(|e| e.to_string())?;
    m.add_resident(&[texture.handle()]);
    let twin = if def.twin {
        let twin = crate::backend::metal_depth::StencilTwin::new(&m.device, &texture, &def.desc, def.mips, def.slices).map_err(|e| e.to_string())?;
        m.add_resident(&twin.handles());
        Some(Arc::new(twin))
    } else {
        None
    };
    Ok(MetalTexture { texture, pixel: WMTPixelFormat(def.pixel), desc: def.desc, mips: def.mips, slices: def.slices, companion: Default::default(), twin })
}

/// What the record side knows of the backend's textures this run.
#[derive(Default)]
pub(crate) struct TexMap {
    /// Metal texture object → (title resource, view).
    pub(crate) by_handle: HashMap<u64, (usize, u64)>,
    pub(crate) textures: HashMap<usize, Arc<MetalTexture>>,
}

impl TexMap {
    pub(crate) fn of(m: &MetalBackend) -> Self {
        let mut out = TexMap::default();
        for (&key, t) in m.textures.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            let Some(t) = t else { continue };
            out.by_handle.insert(t.texture.handle(), (key, VIEW_SELF));
            if let Some(Some(c)) = t.companion.get() {
                out.by_handle.insert(c.texture.handle(), (key, VIEW_COMPANION));
            }
            out.textures.insert(key, t.clone());
        }
        for (&(key, pixel), v) in m.target_views.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            out.by_handle.insert(v.handle(), (key, VIEW_TARGET | u64::from(pixel)));
        }
        out
    }
}

/// Per texture, as last seen.
struct TexState {
    def: TexDef,
    hash: Option<[u8; 32]>,
    written: bool,
    unread: bool,
}

/// Texture ids, definitions and contents of a recording.
pub(crate) struct TextureRecorder {
    ids: HashMap<usize, u32>,
    states: Vec<TexState>,
    defs: Vec<u8>,
    idx: Vec<u8>,
    bin: std::io::BufWriter<std::fs::File>,
    at: u64,
}

/// `textures.idx` / `reference/textures.idx` record: `id u32 | epoch u32 | offset u64 | len u64`.
pub(crate) const TEXTURE_RECORD: usize = 24;
pub(crate) const TEXTURES_MAGIC: &[u8; 8] = b"VPKTEX01";

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TexStats {
    pub(crate) textures: u64,
    pub(crate) versions: u64,
    pub(crate) bytes: u64,
    pub(crate) unread: u64,
    pub(crate) twins: u64,
    pub(crate) reference: u64,
}

impl TextureRecorder {
    pub(crate) fn new(dir: &std::path::Path) -> std::io::Result<Self> {
        Ok(Self {
            ids: HashMap::new(),
            states: Vec::new(),
            defs: TEXTURES_MAGIC.to_vec(),
            idx: Vec::new(),
            bin: std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(dir.join("textures.bin"))?),
            at: 0,
        })
    }

    /// The package id of title texture `key` (defined on first sight).
    pub(crate) fn id(&mut self, key: usize, t: &MetalTexture, stats: &mut TexStats) -> u32 {
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.states.len() as u32;
        let def = TexDef::of(id, key, t);
        def.put(&mut self.defs);
        let unread = def.layout().is_none();
        stats.textures += 1;
        stats.unread += unread as u64;
        stats.twins += def.twin as u64;
        self.states.push(TexState { def, hash: None, written: false, unread });
        self.ids.insert(key, id);
        id
    }

    /// The reference for the texture object `handle` (`Unknown` when the
    /// backend does not know it).
    pub(crate) fn tex(&mut self, map: &TexMap, handle: u64, stats: &mut TexStats) -> Ref {
        match map.by_handle.get(&handle) {
            Some(&(key, view)) => match map.textures.get(&key) {
                Some(t) => Ref::Tex { id: self.id(key, t, stats), view },
                None => Ref::Unknown { raw: handle },
            },
            None => Ref::Unknown { raw: handle },
        }
    }

    /// Before run `run`: the contents of textures `ids` that are new, or
    /// (title-filled ones) changed since; at most `room` bytes of
    /// textures in all. False when the room ran out.
    pub(crate) fn snapshot(&mut self, m: &MetalBackend, map: &TexMap, ids: &BTreeSet<u32>, run: u32, room: u64, stats: &mut TexStats) -> bool {
        for &id in ids {
            let s = &self.states[id as usize];
            if s.unread || (s.hash.is_some() && s.def.writable()) {
                continue;
            }
            let def = s.def;
            let Some(t) = map.textures.get(&(def.key as usize)) else { continue };
            let Some(bytes) = read_texture(m, &t.texture, &def) else {
                self.states[id as usize].unread = true;
                stats.unread += 1;
                continue;
            };
            let hash = *blake3::hash(&bytes).as_bytes();
            if self.states[id as usize].hash == Some(hash) {
                continue;
            }
            if stats.bytes + bytes.len() as u64 > room || self.bin.write_all(&bytes).is_err() {
                return false;
            }
            put_u32(&mut self.idx, id);
            put_u32(&mut self.idx, run);
            put_u64(&mut self.idx, self.at);
            put_u64(&mut self.idx, bytes.len() as u64);
            self.at += bytes.len() as u64;
            stats.bytes += bytes.len() as u64;
            stats.versions += 1;
            self.states[id as usize].hash = Some(hash);
        }
        true
    }

    /// The textures a run may have written.
    pub(crate) fn mark_written(&mut self, ids: &BTreeSet<u32>) {
        for &id in ids {
            if let Some(s) = self.states.get_mut(id as usize) {
                s.written = true;
            }
        }
    }

    /// The contents of every texture the frame may have written go to
    /// `reference/textures.*`; returns the texture files to write.
    pub(crate) fn finish(&mut self, m: Option<&MetalBackend>, dir: &std::path::Path, stats: &mut TexStats) -> std::io::Result<Vec<(&'static str, Vec<u8>)>> {
        self.bin.flush()?;
        let mut ref_idx = Vec::new();
        let mut ref_bin = std::io::BufWriter::new(std::fs::File::create(dir.join("reference").join("textures.bin"))?);
        let mut at = 0u64;
        if let Some(m) = m {
            let map = TexMap::of(m);
            for s in self.states.iter().filter(|s| s.written && !s.unread) {
                let Some(t) = map.textures.get(&(s.def.key as usize)) else { continue };
                let Some(bytes) = read_texture(m, &t.texture, &s.def) else { continue };
                ref_bin.write_all(&bytes)?;
                put_u32(&mut ref_idx, s.def.id);
                put_u32(&mut ref_idx, u32::MAX);
                put_u64(&mut ref_idx, at);
                put_u64(&mut ref_idx, bytes.len() as u64);
                at += bytes.len() as u64;
                stats.reference += 1;
            }
        }
        ref_bin.flush()?;
        Ok(vec![
            ("textures.defs", std::mem::take(&mut self.defs)),
            ("textures.idx", std::mem::take(&mut self.idx)),
            ("reference/textures.idx", ref_idx),
        ])
    }
}

/// Texture ids an op references (into `out`) and may write (`written`).
pub(crate) fn op_textures(op: &super::POp, out: &mut BTreeSet<u32>, written: &mut BTreeSet<u32>) {
    use super::POp;
    let mut tex = |r: &Ref, w: bool| {
        if let Ref::Tex { id, .. } = r {
            out.insert(*id);
            if w {
                written.insert(*id);
            }
        }
    };
    match op {
        POp::Draw(d) => {
            for a in d.targets.colors.iter().flatten().chain(d.targets.depth.iter()) {
                tex(&a.tex, true);
            }
        }
        POp::Clear(c) => tex(&c.target.tex, true),
        POp::Copy(c) => {
            tex(&c.src, false);
            tex(&c.dst, true);
        }
        POp::Resolve(c) => {
            tex(&c.src, false);
            tex(&c.dst, true);
        }
        POp::Readback { src, .. } => tex(src, false),
        POp::Ambient { tex: t, .. } => tex(t, true),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Recording the draw-side ops
// ---------------------------------------------------------------------------

/// What translating draw-side ops needs from the recorder.
pub(crate) struct DrawCtx<'a> {
    pub(crate) map: &'a super::AddressMap,
    pub(crate) texmap: &'a TexMap,
    pub(crate) textures: &'a mut TextureRecorder,
    pub(crate) stats: &'a mut TexStats,
    pub(crate) graphics: &'a mut HashMap<usize, u32>,
    pub(crate) graphics_bin: &'a mut Vec<u8>,
    pub(crate) graphics_without_source: &'a mut u32,
}

impl DrawCtx<'_> {
    fn attachment(&mut self, a: &Attachment) -> PAttachment {
        PAttachment {
            tex: self.textures.tex(self.texmap, a.texture.handle(), self.stats),
            pixel: a.pixel,
            level: a.level,
            slice: a.slice,
            width: a.width,
            height: a.height,
            samples: a.samples,
            companion: a.companion.is_some(),
            twin: a.twin.is_some(),
        }
    }

    fn graphics_id(&mut self, g: &Arc<MetalGraphics>) -> u32 {
        let key = Arc::as_ptr(g) as usize;
        if let Some(&id) = self.graphics.get(&key) {
            return id;
        }
        let id = self.graphics.len() as u32;
        self.graphics.insert(key, id);
        let source = graphics_source(g);
        if source.is_none() {
            *self.graphics_without_source += 1;
        }
        put_graphics(self.graphics_bin, id, g, source.as_deref());
        id
    }

    /// `None` for rectangle emulation (not recorded yet).
    pub(crate) fn draw(&mut self, d: &MetalDraw) -> Option<PDraw> {
        let map = self.map;
        if matches!(d.kind, DrawKind::Geometry { .. }) {
            return None;
        }
        let t = &d.targets;
        let mut targets = PTargets { stencil: t.stencil, width: t.width, height: t.height, samples: t.samples, ..Default::default() };
        for (i, c) in t.colors.iter().enumerate() {
            targets.colors[i] = c.as_ref().map(|a| self.attachment(a));
        }
        targets.depth = t.depth.as_ref().map(|a| self.attachment(a));
        let buffer = |b: &Option<Buffer>| b.as_ref().map_or(Ref::Null, |b| map.address(b.gpu_address, Some(b.len)));
        let stages = d
            .stages
            .iter()
            .map(|s| {
                // The pipeline's stage that binds there gives the layout.
                let (layout, template) = d.pso.stages.iter().find(|(_, b)| *b == s.bind).map_or((&[][..], &[][..]), |(top, _)| (&top.layout[..], &top.bytes[..]));
                PStage {
                    bind: s.bind,
                    top: map.top(layout, template, &s.top, &s.tables),
                    tables: s.tables.iter().map(|(at, e)| (*at, map.entry(e, false))).collect(),
                    heap: buffer(&s.heap),
                    sampler_heap: buffer(&s.sampler_heap),
                }
            })
            .collect();
        let vertex = d.vertex.as_ref().map(|v| {
            v.table
                .chunks_exact(16)
                .map(|e| {
                    let address = u64::from_le_bytes(e[..8].try_into().unwrap());
                    let length = u32::from_le_bytes(e[8..12].try_into().unwrap());
                    let stride = u32::from_le_bytes(e[12..].try_into().unwrap());
                    (map.address(address, Some(u64::from(length))), length, stride)
                })
                .collect()
        });
        let index_bytes = |t: u8| if t == 0 { 2 } else { 4 };
        let kind = match &d.kind {
            DrawKind::Vertices { count, instances, first, first_instance } => {
                PDrawKind::Vertices { count: *count, instances: *instances, first: *first, first_instance: *first_instance }
            }
            DrawKind::Indexed { count, instances, base, first_instance, buffer, offset, index_type } => PDrawKind::Indexed {
                count: *count,
                instances: *instances,
                base: *base,
                first_instance: *first_instance,
                buffer: map.address(buffer.gpu_address + u64::from(*offset), Some(u64::from(*count) * index_bytes(*index_type))),
                index_type: *index_type,
            },
            DrawKind::Mesh { groups } => PDrawKind::Mesh { groups: *groups },
            DrawKind::Generated { indices, instances, base, first_instance } => {
                PDrawKind::Generated { indices: indices.clone(), instances: *instances, base: *base, first_instance: *first_instance }
            }
            DrawKind::Indirect { records, records_offset, records_va, stride, max, count, count_va, index } => PDrawKind::Indirect {
                records: map.address(records.gpu_address + records_offset, Some(u64::from(*stride) * u64::from((*max).max(1)))),
                records_va: *records_va,
                stride: *stride,
                max: *max,
                count: count.as_ref().map(|(b, o)| map.address(b.gpu_address + o, Some(4))),
                count_va: *count_va,
                index: index.as_ref().map(|(b, o, t, cap)| (map.address(b.gpu_address + o, Some(u64::from(*cap) * index_bytes(*t))), *t, *cap)),
            },
            DrawKind::Geometry { .. } => return None,
        };
        let v = &d.viewport;
        let s = &d.scissor;
        Some(PDraw {
            pipeline: self.graphics_id(&d.pso),
            targets,
            stages,
            vertex,
            viewport: [v.originX, v.originY, v.width, v.height, v.znear, v.zfar],
            scissor: [s.x, s.y, s.width, s.height],
            blend: d.blend,
            stencil_ref: d.stencil_ref,
            primitive: d.primitive,
            kind,
        })
    }

    pub(crate) fn clear(&mut self, c: &crate::backend::metal_draw::MetalClear) -> PClear {
        PClear {
            target: self.attachment(&c.target),
            color: c.color,
            depth: c.depth,
            stencil: c.stencil,
            has_stencil: c.has_stencil,
            rects: c.rects.iter().map(|r| [r.left, r.top, r.right, r.bottom]).collect(),
        }
    }

    pub(crate) fn copy(&mut self, c: &crate::backend::metal_draw::MetalTexCopy) -> PCopy {
        PCopy {
            src: self.tex(&c.src.texture),
            dst: self.tex(&c.dst.texture),
            whole: c.whole,
            src_slice: c.src_slice,
            src_level: c.src_level,
            src_origin: c.src_origin,
            size: c.size,
            dst_slice: c.dst_slice,
            dst_level: c.dst_level,
            dst_origin: c.dst_origin,
        }
    }

    pub(crate) fn resolve(&mut self, c: &crate::backend::metal_draw::MetalResolve) -> PResolve {
        PResolve {
            src: self.tex(&c.src),
            src_slice: c.src_slice,
            dst: self.tex(&c.dst),
            dst_level: c.dst_level,
            dst_slice: c.dst_slice,
            format: c.format.0,
            region: c.region.map(|r| [r.src_origin[0], r.src_origin[1], r.dst_origin[0], r.dst_origin[1], r.size[0], r.size[1]]),
        }
    }

    pub(crate) fn tex(&mut self, texture: &Texture) -> Ref {
        self.textures.tex(self.texmap, texture.handle(), self.stats)
    }
}
