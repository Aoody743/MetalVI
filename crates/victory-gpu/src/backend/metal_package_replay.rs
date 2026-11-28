//! Replaying a frame package (F3): title memory put back at its own
//! addresses (no remapping: the memory directory and every title pointer
//! inside buffers need them), heaps and pipelines rebuilt, then each run
//! through [`MetalBackend::run`] in order; the pages the GPU wrote are
//! compared with the package's reference.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use victory_metal::Buffer;
use victory_metal::ir::{DescriptorEntry, TopLevel};
use victory_metal::msc::{RootParam, StaticSampler};

use super::{
    get_params, get_runs, get_samplers, resolve_entry, resolve_top, PAGE, POp, PCompute, Reader, Ref, Resolver, TEntry,
    MEMORY_RECORD, SPECIAL_DIAG, SPECIAL_WAVE_SCRATCH,
};
use super::draw::{self, GraphicsDef, PAttachment, PDraw, PDrawKind, TexDef, VIEW_COMPANION, VIEW_SELF, VIEW_TARGET};
use crate::backend::metal::{self as metal, MetalBackend, MetalCmd, MetalCompute, MetalDispatchIndirect, MetalOp};
use crate::backend::metal_draw::{Attachment, DrawKind, MetalClear, MetalDraw, MetalGraphics, MetalResolve, MetalTexCopy, StageBind, Targets, VertexFetch};
use crate::backend::metal_texture::MetalTexture;
use victory_metal::texture::Texture;

/// One stored texture version (`textures.idx`, `reference/textures.idx`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TexRecord {
    pub(crate) id: u32,
    pub(crate) epoch: u32,
    pub(crate) offset: u64,
    pub(crate) len: u64,
}

fn tex_records(b: &[u8], name: &str) -> Result<Vec<TexRecord>, String> {
    if !b.len().is_multiple_of(draw::TEXTURE_RECORD) {
        return Err(format!("{name}: not a whole number of records"));
    }
    let mut r = Reader::new(b);
    let mut out = Vec::new();
    while !r.done() {
        let (Some(id), Some(epoch), Some(offset), Some(len)) = (r.u32(), r.u32(), r.u64(), r.u64()) else {
            return Err(format!("{name}: truncated"));
        };
        out.push(TexRecord { id, epoch, offset, len });
    }
    Ok(out)
}

/// One stored page (`memory.idx`, `reference/memory.idx`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemRecord {
    pub(crate) va: u64,
    pub(crate) len: u32,
    /// The run before which the content applies (`u32::MAX`: reference).
    pub(crate) epoch: u32,
    pub(crate) offset: u64,
}

pub(crate) enum DefSource {
    Dxil { dxil: Vec<u8>, params: Vec<RootParam>, samplers: Vec<StaticSampler>, vmsc: Option<Vec<u8>> },
    Msl { source: String, function: String },
}

pub(crate) struct PipelineDef {
    pub(crate) id: u32,
    pub(crate) name: String,
    pub(crate) threadgroup: [u64; 3],
    pub(crate) layout: Vec<u8>,
    pub(crate) template: Vec<u8>,
    pub(crate) source: Option<DefSource>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct HeapDef {
    pub(crate) id: u32,
    pub(crate) count: u32,
    pub(crate) capacity: u32,
    pub(crate) samplers: bool,
}

pub(crate) enum HeapUpdate {
    Entries { id: u32, run: u32, entries: Vec<(u32, TEntry)> },
    Strides { id: u32, run: u32, words: Vec<(u32, u32)> },
}

/// A frame package read back.
pub(crate) struct Package {
    pub(crate) dir: PathBuf,
    pub(crate) pipelines: Vec<PipelineDef>,
    pub(crate) runs: Vec<(u32, Vec<POp>)>,
    pub(crate) memory: Vec<MemRecord>,
    pub(crate) reference: Vec<MemRecord>,
    pub(crate) heaps: Vec<HeapDef>,
    pub(crate) updates: Vec<HeapUpdate>,
    pub(crate) samplers: HashMap<u32, [u32; 13]>,
    pub(crate) typed: HashMap<u32, (u64, u64, u32)>,
    pub(crate) textures: HashMap<u32, (u64, [u32; 10], bool)>,
    pub(crate) graphics: Vec<GraphicsDef>,
    pub(crate) texture_defs: Vec<TexDef>,
    pub(crate) texture_versions: Vec<TexRecord>,
    pub(crate) texture_reference: Vec<TexRecord>,
}

fn read(dir: &Path, name: &str) -> Result<Vec<u8>, String> {
    crate::read_file(&dir.join(name)).map_err(|e| format!("{name}: {e}"))
}

fn records(b: &[u8], name: &str) -> Result<Vec<MemRecord>, String> {
    if !b.len().is_multiple_of(MEMORY_RECORD) {
        return Err(format!("{name}: {} bytes is not a whole number of records", b.len()));
    }
    let mut r = Reader::new(b);
    let mut out = Vec::with_capacity(b.len() / MEMORY_RECORD);
    while !r.done() {
        let (va, len, epoch, offset, flags) = (r.u64(), r.u32(), r.u32(), r.u64(), r.u32());
        let (Some(va), Some(len), Some(epoch), Some(offset), Some(flags)) = (va, len, epoch, offset, flags) else {
            return Err(format!("{name}: truncated"));
        };
        if flags != 0 {
            return Err(format!("{name}: flags {flags:#x} (compressed pages) are not read by this build"));
        }
        out.push(MemRecord { va, len, epoch, offset });
    }
    Ok(out)
}

impl Package {
    pub(crate) fn open(dir: &Path) -> Result<Self, String> {
        let runs = get_runs(&read(dir, "ops.bin")?)?;
        let memory = records(&read(dir, "memory.idx")?, "memory.idx")?;
        let reference = records(&read(dir, "reference/memory.idx")?, "reference/memory.idx")?;

        let b = read(dir, "pipelines.bin")?;
        let mut r = Reader::new(&b);
        if r.bytes(8) != Some(b"VPKPIP01".as_slice()) {
            return Err("pipelines.bin: bad magic".into());
        }
        let bad = || "pipelines.bin: truncated".to_string();
        let n = r.u32().ok_or_else(bad)?;
        let mut pipelines = Vec::new();
        for _ in 0..n {
            let id = r.u32().ok_or_else(bad)?;
            let name = r.text().ok_or_else(bad)?;
            let threadgroup = [r.u64().ok_or_else(bad)?, r.u64().ok_or_else(bad)?, r.u64().ok_or_else(bad)?];
            let layout = r.blob().ok_or_else(bad)?.to_vec();
            let template = r.blob().ok_or_else(bad)?.to_vec();
            let source = match r.u8().ok_or_else(bad)? {
                0 => None,
                1 => {
                    let file = r.text().ok_or_else(bad)?;
                    let params = get_params(&mut r).ok_or_else(bad)?;
                    let samplers = get_samplers(&mut r).ok_or_else(bad)?;
                    let vmsc = r.text().ok_or_else(bad)?;
                    let dxil = read(dir, &format!("shaders/{file}"))?;
                    let vmsc = if vmsc.is_empty() { None } else { read(dir, &format!("shaders/{vmsc}")).ok() };
                    Some(DefSource::Dxil { dxil, params, samplers, vmsc })
                }
                2 => Some(DefSource::Msl { source: r.text().ok_or_else(bad)?, function: r.text().ok_or_else(bad)? }),
                k => return Err(format!("pipelines.bin: source kind {k}")),
            };
            pipelines.push(PipelineDef { id, name, threadgroup, layout, template, source });
        }

        let b = read(dir, "heaps.bin")?;
        let mut r = Reader::new(&b);
        if r.bytes(8) != Some(b"VPKHEP01".as_slice()) {
            return Err("heaps.bin: bad magic".into());
        }
        let bad = || "heaps.bin: truncated".to_string();
        let (mut heaps, mut updates) = (Vec::new(), Vec::new());
        while !r.done() {
            match r.u8().ok_or_else(bad)? {
                1 => heaps.push(HeapDef {
                    id: r.u32().ok_or_else(bad)?,
                    count: r.u32().ok_or_else(bad)?,
                    capacity: r.u32().ok_or_else(bad)?,
                    samplers: r.u8().ok_or_else(bad)? != 0,
                }),
                2 => {
                    let (id, run, n) = (r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?);
                    let mut entries = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        entries.push((r.u32().ok_or_else(bad)?, TEntry::get(&mut r).ok_or_else(bad)?));
                    }
                    updates.push(HeapUpdate::Entries { id, run, entries });
                }
                3 => {
                    let (id, run, n) = (r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?);
                    let mut words = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        words.push((r.u32().ok_or_else(bad)?, r.u32().ok_or_else(bad)?));
                    }
                    updates.push(HeapUpdate::Strides { id, run, words });
                }
                t => return Err(format!("heaps.bin: record {t}")),
            }
        }

        let b = read(dir, "resources.bin")?;
        let mut r = Reader::new(&b);
        if r.bytes(8) != Some(b"VPKRES01".as_slice()) {
            return Err("resources.bin: bad magic".into());
        }
        let bad = || "resources.bin: truncated".to_string();
        let (mut samplers, mut typed, mut textures) = (HashMap::new(), HashMap::new(), HashMap::new());
        while !r.done() {
            let tag = r.u8().ok_or_else(bad)?;
            let id = r.u32().ok_or_else(bad)?;
            match tag {
                1 => {
                    let mut w = [0u32; 13];
                    for x in &mut w {
                        *x = r.u32().ok_or_else(bad)?;
                    }
                    samplers.insert(id, w);
                }
                2 => {
                    typed.insert(id, (r.u64().ok_or_else(bad)?, r.u64().ok_or_else(bad)?, r.u32().ok_or_else(bad)?));
                }
                3 => {
                    let resource = r.u64().ok_or_else(bad)?;
                    let mut w = [0u32; 10];
                    for x in &mut w {
                        *x = r.u32().ok_or_else(bad)?;
                    }
                    textures.insert(id, (resource, w, r.u8().ok_or_else(bad)? != 0));
                }
                t => return Err(format!("resources.bin: record {t}")),
            }
        }
        // F4 files (a compute-only package of F3 has none).
        let mut graphics = Vec::new();
        if let Ok(b) = crate::read_file(&dir.join("graphics.bin")) {
            let mut r = Reader::new(&b);
            if r.bytes(8) != Some(b"VPKGFX01".as_slice()) {
                return Err("graphics.bin: bad magic".into());
            }
            let n = r.u32().ok_or("graphics.bin: truncated")?;
            for _ in 0..n {
                graphics.push(draw::get_graphics(&mut r).ok_or("graphics.bin: truncated")?);
            }
        }
        let mut texture_defs = Vec::new();
        if let Ok(b) = crate::read_file(&dir.join("textures.defs")) {
            let mut r = Reader::new(&b);
            if r.bytes(8) != Some(draw::TEXTURES_MAGIC.as_slice()) {
                return Err("textures.defs: bad magic".into());
            }
            while !r.done() {
                texture_defs.push(TexDef::get(&mut r).ok_or("textures.defs: truncated")?);
            }
        }
        let texture_versions = crate::read_file(&dir.join("textures.idx")).map_or(Ok(Vec::new()), |b| tex_records(&b, "textures.idx"))?;
        let texture_reference =
            crate::read_file(&dir.join("reference").join("textures.idx")).map_or(Ok(Vec::new()), |b| tex_records(&b, "reference/textures.idx"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            pipelines,
            runs,
            memory,
            reference,
            heaps,
            updates,
            samplers,
            typed,
            textures,
            graphics,
            texture_defs,
            texture_versions,
            texture_reference,
        })
    }

    /// The page-aligned ranges a replay needs at their own addresses
    /// (every stored page, merged where they touch).
    pub(crate) fn ranges(&self) -> Vec<(u64, u64)> {
        let mut pages: Vec<u64> = self.memory.iter().chain(&self.reference).map(|m| m.va & !(PAGE - 1)).collect();
        pages.sort_unstable();
        pages.dedup();
        let mut out: Vec<(u64, u64)> = Vec::new();
        for p in pages {
            match out.last_mut() {
                Some((s, n)) if *s + *n == p => *n += PAGE,
                _ => out.push((p, PAGE)),
            }
        }
        out
    }

    fn page_bytes(&self, file: &mut std::fs::File, m: &MemRecord) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; m.len as usize];
        file.seek(SeekFrom::Start(m.offset)).and_then(|_| file.read_exact(&mut bytes)).map_err(|e| format!("page {:#x}: {e}", m.va))?;
        Ok(bytes)
    }
}

/// What is already in this process where the package's memory must go:
/// (start, bytes, what) per occupied region.
pub(crate) fn conflicts(ranges: &[(u64, u64)]) -> Vec<(u64, u64, String)> {
    use windows_sys::Win32::System::Memory::{MEM_FREE, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut out = Vec::new();
    for &(start, len) in ranges {
        let mut at = start;
        while at < start + len {
            // SAFETY: plain data filled in by the query.
            let mut m: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            if unsafe { VirtualQuery(at as *const core::ffi::c_void, &mut m, size_of_val(&m)) } == 0 {
                out.push((at, start + len - at, "not queryable (outside the user address space?)".into()));
                break;
            }
            let end = (m.BaseAddress as u64 + m.RegionSize as u64).min(start + len);
            if m.State != MEM_FREE {
                out.push((at, end - at, format!("state {:#x} type {:#x} protect {:#x} (allocation at {:#x})", m.State, m.Type, m.Protect, m.AllocationBase as u64)));
            }
            at = end;
        }
    }
    out
}

/// Reserves and commits the package's ranges at their own addresses. Fails,
/// naming every conflicting region, if any of them is taken: nothing is
/// remapped. The memory stays for the life of the process.
pub(crate) fn place(pkg: &Package) -> Result<Vec<(u64, u64)>, String> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    let ranges = pkg.ranges();
    let taken = conflicts(&ranges);
    if !taken.is_empty() {
        let list: Vec<String> = taken.iter().map(|(s, n, what)| format!("[{s:#x}, {:#x}) {what}", s + n)).collect();
        return Err(format!("{} title ranges are taken in this process:\n{}", taken.len(), list.join("\n")));
    }
    for &(start, len) in &ranges {
        // SAFETY: the range is free (checked above); a fixed allocation there.
        let p = unsafe { VirtualAlloc(start as *const core::ffi::c_void, len as usize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if p as u64 != start {
            return Err(format!("VirtualAlloc({start:#x}, {len:#x}) gave {:#x}", p as u64));
        }
    }
    Ok(ranges)
}

/// Writes the pages of `epoch` into place.
fn write_epoch(pkg: &Package, file: &mut std::fs::File, epoch: u32) -> Result<usize, String> {
    let mut n = 0;
    for m in pkg.memory.iter().filter(|m| m.epoch == epoch) {
        let bytes = pkg.page_bytes(file, m)?;
        // SAFETY: placed by `place` (every stored page is in its ranges).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), m.va as *mut u8, bytes.len()) };
        n += 1;
    }
    Ok(n)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Options {
    /// Only runs `a..b` go to the GPU (memory and heaps still follow every run).
    pub(crate) runs: Option<(u32, u32)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Report {
    pub(crate) runs: u32,
    pub(crate) ops_run: u64,
    pub(crate) pipelines_built: u32,
    pub(crate) pipeline_errors: Vec<String>,
    pub(crate) layout_mismatches: u32,
    pub(crate) skipped_unsupported: u64,
    pub(crate) skipped_no_pipeline: u64,
    pub(crate) skipped_memory_directory: u64,
    pub(crate) skipped_unmapped: u64,
    pub(crate) zeroed_unknown: u64,
    pub(crate) zeroed_texture: u64,
    pub(crate) unmapped_title: u64,
    pub(crate) heap_capacity_mismatches: u32,
    pub(crate) pages_written: u64,
    pub(crate) failed_run: Option<u32>,
    pub(crate) graphics_built: u32,
    pub(crate) graphics_errors: Vec<String>,
    pub(crate) textures_made: u32,
    pub(crate) texture_errors: Vec<String>,
    pub(crate) texture_uploads: u64,
    /// Texture references that could not be had (draw-side ops skipped).
    pub(crate) skipped_texture: u64,
}

/// Fake title-heap keys for the replayed heaps in `MetalBackend::heaps`.
const HEAP_KEY: usize = 0x7f9a_0000_0000;

struct Replayer<'a> {
    m: &'a MetalBackend,
    pkg: &'a Package,
    heaps: HashMap<u32, (Buffer, Option<Buffer>)>,
    textures: HashMap<u32, Arc<MetalTexture>>,
    nulls: HashMap<(u32, u32), Texture>,
    report: Report,
}

impl Resolver for Replayer<'_> {
    fn address(&mut self, r: &Ref) -> u64 {
        match *r {
            Ref::Null | Ref::RingTable | Ref::Template => 0,
            Ref::Title { va, len } => {
                let gpu = self.m.chunks().window(va, u64::from(len.max(1)));
                if gpu == 0 {
                    self.report.unmapped_title += 1;
                }
                gpu
            }
            Ref::Heap { id, offset } => self.heaps.get(&id).map_or(0, |(b, _)| b.gpu_address + offset),
            Ref::HeapStrides { id, offset } => self.heaps.get(&id).and_then(|(_, s)| s.as_ref()).map_or(0, |s| s.gpu_address + offset),
            Ref::Sampler { id } => self.pkg.samplers.get(&id).and_then(|w| self.m.sampler_entry(w)).map_or(0, |e| e.gpu_va),
            Ref::Typed { id } => self.typed(id).map_or(0, |e| e.texture_view_id),
            Ref::Special { kind: SPECIAL_WAVE_SCRATCH, offset } => self.m.wave_scratch_gpu() + offset,
            Ref::Special { kind: SPECIAL_DIAG, offset } => metal::diag::gpu(self.m) + offset,
            Ref::Texture { id } => match self.heap_view(id) {
                Some(v) => v,
                None => {
                    self.report.zeroed_texture += 1;
                    0
                }
            },
            Ref::Tex { .. } => self.texture(r).map_or(0, |t| t.gpu_resource_id()),
            Ref::Special { .. } | Ref::Unknown { .. } => {
                self.report.zeroed_unknown += 1;
                0
            }
        }
    }

    fn typed(&mut self, id: u32) -> Option<DescriptorEntry> {
        let &(va, bytes, format) = self.pkg.typed.get(&id)?;
        self.m.typed_entry(va, bytes, format)
    }
}

impl Replayer<'_> {
    /// The `gpuResourceID` of heap texture view `id`, made anew on the
    /// rebuilt texture (or a null texture's, for resource 0).
    fn heap_view(&mut self, id: u32) -> Option<u64> {
        let &(resource, words, unordered) = self.pkg.textures.get(&id)?;
        if resource == 0 {
            return self.null_view(words[0], words[1]);
        }
        let n = words.iter().rposition(|&w| w != u32::MAX).map_or(0, |i| i + 1);
        let words = (n > 0).then(|| &words[..n]);
        self.m.texture_entry(resource as crate::object::This, words, unordered).map(|e| e.texture_view_id)
    }

    /// A zeroed 1x1 texture of Metal type `type_` and format `pixel`, as the
    /// backend's null views are.
    fn null_view(&mut self, type_: u32, pixel: u32) -> Option<u64> {
        use victory_metal::sys::{WMTPixelFormat, WMTTextureType, WMTTextureUsage};
        if let Some(t) = self.nulls.get(&(type_, pixel)) {
            return Some(t.gpu_resource_id());
        }
        let cube = type_ == WMTTextureType::WMTTextureTypeCube.0 || type_ == WMTTextureType::WMTTextureTypeCubeArray.0;
        let t = self
            .m
            .device
            .new_texture(&victory_metal::texture::TextureDescriptor {
                pixel_format: WMTPixelFormat(pixel),
                type_: WMTTextureType(type_),
                usage: WMTTextureUsage(1 | 2),
                options: victory_metal::sys::WMTResourceOptions(0x20),
                ..Default::default()
            })
            .ok()?;
        let zeros = self.m.device.shared_buffer(&[0; 256]).ok()?;
        let cb = self.m.queue.command_buffer().ok()?;
        let mut blit = cb.blit_encoder().ok()?;
        for slice in 0..if cube { 6 } else { 1 } {
            blit.copy_from_buffer_to_texture(&zeros, 0, 256, 256, [1, 1, 1], &t, slice, 0, [0; 3]);
        }
        blit.finish();
        metal::submit_checked(&cb, &[], true, "frame package null texture").ok()?;
        self.m.add_resident(&[t.handle()]);
        let id = t.gpu_resource_id();
        self.nulls.insert((type_, pixel), t);
        Some(id)
    }

    fn metal_texture(&self, r: &Ref) -> Option<Arc<MetalTexture>> {
        match r {
            Ref::Tex { id, .. } => self.textures.get(id).cloned(),
            _ => None,
        }
    }

    /// The Metal texture a [`Ref::Tex`] names: the texture, a target view
    /// of it (made as `MetalBackend::attachment` makes it), its companion.
    fn texture(&mut self, r: &Ref) -> Option<Texture> {
        let &Ref::Tex { id, view } = r else { return None };
        let t = self.textures.get(&id)?.clone();
        match view {
            VIEW_SELF => Some(t.texture.clone()),
            VIEW_COMPANION => t.depth_companion(&self.m.device, 0).map(|c| c.texture),
            v if v & !0xffff_ffff == VIEW_TARGET => {
                let pixel = (v & 0xffff_ffff) as u32;
                let key = (self.pkg.texture_defs.get(id as usize)?.key as usize, pixel);
                let mut views = self.m.target_views.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(v) = views.get(&key) {
                    return Some(v.clone());
                }
                let _pool = victory_metal::Pool::new();
                let mut gpu_id = 0u64;
                // SAFETY: a live texture; every level and slice, its own type.
                let made = unsafe {
                    victory_metal::sys::MTLTexture_newTextureView(
                        t.texture.handle(),
                        victory_metal::sys::WMTPixelFormat(pixel),
                        crate::backend::metal_texture::base_type(&t),
                        0,
                        t.mips as u16,
                        0,
                        t.slices.max(1) as u16,
                        crate::backend::metal_texture::identity(),
                        &mut gpu_id,
                    )
                };
                let view = Texture::from_object(victory_metal::Object::adopt(made)?);
                views.insert(key, view.clone());
                Some(view)
            }
            _ => None,
        }
    }

    fn attachment(&mut self, a: &PAttachment) -> Option<Attachment> {
        let t = self.metal_texture(&a.tex)?;
        let texture = self.texture(&a.tex)?;
        let &Ref::Tex { id, .. } = &a.tex else { return None };
        Some(Attachment {
            resource: self.pkg.texture_defs.get(id as usize).map_or(0, |d| d.key as usize),
            texture,
            pixel: a.pixel,
            level: a.level,
            slice: a.slice,
            width: a.width,
            height: a.height,
            samples: a.samples,
            companion: if a.companion { t.depth_companion(&self.m.device, 0) } else { None },
            twin: if a.twin { t.twin.clone() } else { None },
        })
    }

    fn window(&self, r: &Ref) -> Option<(Buffer, u64)> {
        match *r {
            Ref::Title { va, len } => self.m.chunks().window_buffer(va, u64::from(len.max(1))),
            _ => None,
        }
    }

    fn draw(&mut self, d: &PDraw, graphics: &[Option<Arc<MetalGraphics>>]) -> Option<MetalOp> {
        let Some(pso) = graphics.get(d.pipeline as usize).cloned().flatten() else {
            self.report.skipped_no_pipeline += 1;
            return None;
        };
        let mut targets = Targets { stencil: d.targets.stencil, width: d.targets.width, height: d.targets.height, samples: d.targets.samples, ..Default::default() };
        for (i, c) in d.targets.colors.iter().enumerate() {
            if let Some(a) = c {
                let Some(a) = self.attachment(a) else {
                    self.report.skipped_texture += 1;
                    return None;
                };
                targets.colors[i] = Some(a);
            }
        }
        if let Some(a) = &d.targets.depth {
            let Some(a) = self.attachment(a) else {
                self.report.skipped_texture += 1;
                return None;
            };
            targets.depth = Some(a);
        }
        let mut stages = Vec::new();
        for s in &d.stages {
            if s.top.refs.iter().any(|(_, r)| matches!(r, Ref::Special { kind: super::SPECIAL_MEMORY_DIRECTORY | super::SPECIAL_FAULTS, .. })) {
                self.report.skipped_memory_directory += 1;
                return None;
            }
            let template = pso.stages.iter().find(|(_, b)| *b == s.bind).map(|(t, _)| t.bytes.clone()).unwrap_or_default();
            let top = resolve_top(self, &s.top, &template);
            let tables = s.tables.iter().map(|(at, e)| (*at, resolve_entry(self, e))).collect();
            stages.push(StageBind { bind: s.bind, top, tables, heap: self.heap_buffer(s.heap), sampler_heap: self.heap_buffer(s.sampler_heap) });
        }
        let vertex = d.vertex.as_ref().map(|v| {
            let mut table = [0u8; victory_metal::msc::VERTEX_TABLE_SLOTS * 16];
            let mut buffers = Vec::new();
            for ((r, length, stride), out) in v.iter().zip(table.chunks_exact_mut(16)) {
                let address = match self.window(r) {
                    Some((b, o)) => {
                        let a = b.gpu_address + o;
                        buffers.push(b);
                        a
                    }
                    None => {
                        if matches!(r, Ref::Unknown { .. }) {
                            self.report.zeroed_unknown += 1;
                        }
                        0
                    }
                };
                out[..8].copy_from_slice(&address.to_le_bytes());
                out[8..12].copy_from_slice(&length.to_le_bytes());
                out[12..].copy_from_slice(&stride.to_le_bytes());
            }
            Box::new(VertexFetch { table, _buffers: buffers })
        });
        let kind = match &d.kind {
            PDrawKind::Vertices { count, instances, first, first_instance } => {
                DrawKind::Vertices { count: *count, instances: *instances, first: *first, first_instance: *first_instance }
            }
            PDrawKind::Indexed { count, instances, base, first_instance, buffer, index_type } => {
                let Some((b, o)) = self.window(buffer).filter(|(_, o)| *o <= u64::from(u32::MAX)) else {
                    self.report.skipped_unmapped += 1;
                    return None;
                };
                DrawKind::Indexed { count: *count, instances: *instances, base: *base, first_instance: *first_instance, buffer: b, offset: o as u32, index_type: *index_type }
            }
            PDrawKind::Mesh { groups } => DrawKind::Mesh { groups: *groups },
            PDrawKind::Generated { indices, instances, base, first_instance } => {
                DrawKind::Generated { indices: indices.clone(), instances: *instances, base: *base, first_instance: *first_instance }
            }
            PDrawKind::Indirect { records, records_va, stride, max, count, count_va, index } => {
                let Some((records, records_offset)) = self.window(records) else {
                    self.report.skipped_unmapped += 1;
                    return None;
                };
                let count = match count {
                    Some(c) => Some(self.window(c).or_else(|| {
                        self.report.skipped_unmapped += 1;
                        None
                    })?),
                    None => None,
                };
                let index = match index {
                    Some((r, t, cap)) => {
                        let (b, o) = self.window(r).or_else(|| {
                            self.report.skipped_unmapped += 1;
                            None
                        })?;
                        Some((b, o, *t, *cap))
                    }
                    None => None,
                };
                DrawKind::Indirect { records, records_offset, records_va: *records_va, stride: *stride, max: *max, count, count_va: *count_va, index }
            }
        };
        let v = d.viewport;
        let s = d.scissor;
        Some(MetalOp::Draw(MetalDraw {
            trace_inputs: Vec::new(),
            pso,
            targets: Arc::new(targets),
            stages,
            vertex,
            viewport: victory_metal::sys::WMTViewport { originX: v[0], originY: v[1], width: v[2], height: v[3], znear: v[4], zfar: v[5] },
            scissor: victory_metal::sys::WMTScissorRect { x: s[0], y: s[1], width: s[2], height: s[3] },
            blend: d.blend,
            stencil_ref: d.stencil_ref,
            primitive: d.primitive,
            kind,
        }))
    }

    fn graphics(&mut self, def: &GraphicsDef) -> Option<Arc<MetalGraphics>> {
        match draw::build_graphics(self.m, def) {
            Ok(g) => {
                self.report.graphics_built += 1;
                Some(Arc::new(g))
            }
            Err(e) => {
                self.report.graphics_errors.push(format!("{} ({}): {e}", def.id, def.name));
                None
            }
        }
    }

    fn pipeline(&mut self, def: &PipelineDef) -> Option<Arc<MetalCompute>> {
        let made = match def.source.as_ref() {
            None => Err("no source recorded".to_string()),
            Some(DefSource::Dxil { dxil, params, samplers, vmsc }) => {
                let converted = vmsc.as_deref().and_then(victory_metal::msc::Converted::from_bytes);
                let made = match converted {
                    Some(c) => self.m.device.compute_pipeline_converted(&c).map(|p| (p, c)),
                    None => self.m.device.compute_pipeline_dxil(dxil, params, samplers),
                };
                made.map_err(|e| e.to_string()).and_then(|(pso, mut c)| {
                    self.m.bind_static_samplers(&mut c.top_level, params.len(), samplers)?;
                    Ok(MetalCompute { cost_ns: Default::default(), name: def.name.clone(), pso, threadgroup: c.threadgroup.map(u64::from), top_level: c.top_level })
                })
            }
            Some(DefSource::Msl { source, function }) => {
                self.m.device.compute_pipeline_msl(source, function).map_err(|e| e.to_string()).and_then(|pso| {
                    let (mut top, _) = TopLevel::from_layout_bytes(&def.layout).ok_or("bad layout")?;
                    if top.bytes.len() == def.template.len() {
                        top.bytes.clone_from(&def.template);
                    }
                    Ok(MetalCompute { cost_ns: Default::default(), name: def.name.clone(), pso, threadgroup: def.threadgroup, top_level: top })
                })
            }
        };
        match made {
            Ok(p) => {
                if p.top_level.layout_bytes() != def.layout {
                    self.report.layout_mismatches += 1;
                }
                self.report.pipelines_built += 1;
                Some(Arc::new(p))
            }
            Err(e) => {
                self.report.pipeline_errors.push(format!("{} ({}): {e}", def.id, def.name));
                None
            }
        }
    }

    fn heap_buffer(&self, r: Ref) -> Option<Buffer> {
        match r {
            Ref::Heap { id, .. } => self.heaps.get(&id).map(|(b, _)| b.clone()),
            _ => None,
        }
    }

    fn apply_updates(&mut self, run: u32) {
        let pkg = self.pkg;
        let resolved: Vec<(u32, Vec<(u32, DescriptorEntry)>, Vec<(u32, u32)>)> = pkg
            .updates
            .iter()
            .filter_map(|u| match u {
                HeapUpdate::Entries { id, run: r, entries } if *r == run => {
                    Some((*id, entries.iter().map(|(i, e)| (*i, resolve_entry(self, e))).collect(), Vec::new()))
                }
                HeapUpdate::Strides { id, run: r, words } if *r == run => Some((*id, Vec::new(), words.clone())),
                _ => None,
            })
            .collect();
        let heaps = self.m.heaps.lock().unwrap_or_else(|e| e.into_inner());
        for (id, entries, words) in resolved {
            let Some(h) = heaps.get(&(HEAP_KEY + id as usize)) else { continue };
            for (i, e) in entries {
                if i < h.capacity {
                    // SAFETY: inside the heap's `capacity` entries; the GPU is idle
                    // between runs (each run is waited for).
                    unsafe { h.entries.add(i as usize).write(e) };
                }
            }
            if let Some((_, ptr)) = h.strides {
                for (i, w) in words {
                    if i < victory_shader::emit::SLOT_STRIDES_DWORDS {
                        // SAFETY: inside the stride table (`new_heap`).
                        unsafe { ptr.add(i as usize).write(w) };
                    }
                }
            }
        }
    }

    fn compute(&mut self, c: &PCompute, pipelines: &[Option<Arc<MetalCompute>>]) -> Option<(Arc<MetalCompute>, Vec<u8>, Vec<(u32, DescriptorEntry)>)> {
        let Some(pso) = pipelines.get(c.pipeline as usize).cloned().flatten() else {
            self.report.skipped_no_pipeline += 1;
            return None;
        };
        if c.top.refs.iter().any(|(_, r)| matches!(r, Ref::Special { kind: super::SPECIAL_MEMORY_DIRECTORY | super::SPECIAL_FAULTS, .. })) {
            self.report.skipped_memory_directory += 1;
            return None;
        }
        let top = resolve_top(self, &c.top, &pso.top_level.bytes);
        let tables = c.tables.iter().map(|(at, e)| (*at, resolve_entry(self, e))).collect();
        Some((pso, top, tables))
    }

    fn op(&mut self, op: &POp, pipelines: &[Option<Arc<MetalCompute>>], graphics: &[Option<Arc<MetalGraphics>>]) -> Option<MetalOp> {
        match op {
            POp::Dispatch { c, groups, after_barrier, loop_counts } => {
                let (pso, top, tables) = self.compute(c, pipelines)?;
                let loop_counts = loop_counts.and_then(|vas| {
                    let mut chunks = self.m.chunks();
                    let mut at = |va: u64| chunks.window_buffer(va, 4).map(|(b, o)| (va, b, o));
                    Some([at(vas[0])?, at(vas[1])?])
                });
                Some(MetalOp::Dispatch(MetalCmd {
                    pso,
                    heap: self.heap_buffer(c.heap),
                    sampler_heap: self.heap_buffer(c.sampler_heap),
                    top,
                    groups: *groups,
                    tables,
                    loop_counts,
                    after_barrier: *after_barrier,
                }))
            }
            POp::DispatchIndirect { c, records, stride, max, count, records_va, count_va, group_base_at } => {
                let window = |r: &Ref, m: &MetalBackend| match *r {
                    Ref::Title { va, len } => m.chunks().window_buffer(va, u64::from(len.max(1))),
                    _ => None,
                };
                let Some((records, records_offset)) = window(records, self.m) else {
                    self.report.skipped_unmapped += 1;
                    return None;
                };
                let count = match count {
                    Some(r) => match window(r, self.m) {
                        Some(w) => Some(w),
                        None => {
                            self.report.skipped_unmapped += 1;
                            return None;
                        }
                    },
                    None => None,
                };
                let (pso, top, tables) = self.compute(c, pipelines)?;
                Some(MetalOp::DispatchIndirect(MetalDispatchIndirect {
                    pso,
                    heap: self.heap_buffer(c.heap),
                    sampler_heap: self.heap_buffer(c.sampler_heap),
                    top,
                    tables,
                    records,
                    records_offset,
                    stride: *stride,
                    max: *max,
                    count,
                    records_va: *records_va,
                    count_va: *count_va,
                    group_base_at: group_base_at.map(|v| v as usize),
                }))
            }
            POp::BufferCopy { dst, src, bytes } => Some(MetalOp::BufferCopy { dst: *dst as usize, src: *src as usize, bytes: *bytes as usize }),
            POp::Draw(d) => self.draw(d, graphics),
            POp::Clear(c) => {
                let Some(target) = self.attachment(&c.target) else {
                    self.report.skipped_texture += 1;
                    return None;
                };
                Some(MetalOp::Clear(MetalClear {
                    target,
                    color: c.color,
                    depth: c.depth,
                    stencil: c.stencil,
                    has_stencil: c.has_stencil,
                    rects: c.rects.iter().map(|r| victory_metal::clear_resolve::ClearRect { left: r[0], top: r[1], right: r[2], bottom: r[3] }).collect(),
                }))
            }
            POp::Copy(c) => {
                let (Some(src), Some(dst)) = (self.metal_texture(&c.src), self.metal_texture(&c.dst)) else {
                    self.report.skipped_texture += 1;
                    return None;
                };
                Some(MetalOp::Copy(MetalTexCopy {
                    src,
                    dst,
                    whole: c.whole,
                    src_slice: c.src_slice,
                    src_level: c.src_level,
                    src_origin: c.src_origin,
                    size: c.size,
                    dst_slice: c.dst_slice,
                    dst_level: c.dst_level,
                    dst_origin: c.dst_origin,
                }))
            }
            POp::Readback { src, dst, width } => {
                let (Some(src), Some((dst, offset))) = (self.texture(src), self.window(dst)) else {
                    self.report.skipped_texture += 1;
                    return None;
                };
                Some(MetalOp::FloatRowReadback { src, dst, offset, width: *width })
            }
            POp::Resolve(c) => {
                let (Some(src), Some(dst)) = (self.texture(&c.src), self.texture(&c.dst)) else {
                    self.report.skipped_texture += 1;
                    return None;
                };
                Some(MetalOp::Resolve(MetalResolve {
                    src,
                    src_slice: c.src_slice,
                    dst,
                    dst_level: c.dst_level,
                    dst_slice: c.dst_slice,
                    format: victory_metal::sys::WMTPixelFormat(c.format),
                    region: c.region.map(|r| victory_metal::clear_resolve::ResolveRegion { src_origin: [r[0], r[1]], dst_origin: [r[2], r[3]], size: [r[4], r[5]] }),
                }))
            }
            POp::Ambient { tex, mip, slice, size, colour } => {
                const UAV_TEXTURE2D_ARRAY: u32 = 5;
                let made = self.metal_texture(tex).and_then(|t| crate::backend::metal_ambient::MetalAmbient::new(&t, Some(&[0, UAV_TEXTURE2D_ARRAY, *mip, *slice, 1]), 1.0));
                let Some(mut a) = made.filter(|a| a.size == *size) else {
                    self.report.skipped_texture += 1;
                    return None;
                };
                a.colour = *colour;
                Some(MetalOp::Ambient(a))
            }
            POp::Unsupported { .. } => {
                self.report.skipped_unsupported += 1;
                None
            }
        }
    }
}

/// Replays `pkg` on `m`; its memory must have been placed ([`place`]).
pub(crate) fn replay(m: &MetalBackend, pkg: &Package, opts: &Options) -> Result<Report, String> {
    let _pool = victory_metal::Pool::new();
    let mut r = Replayer { m, pkg, heaps: HashMap::new(), textures: HashMap::new(), nulls: HashMap::new(), report: Report::default() };
    let mut pipelines: Vec<Option<Arc<MetalCompute>>> = Vec::new();
    for def in &pkg.pipelines {
        let p = r.pipeline(def);
        let at = def.id as usize;
        if pipelines.len() <= at {
            pipelines.resize(at + 1, None);
        }
        pipelines[at] = p;
    }
    let mut graphics: Vec<Option<Arc<MetalGraphics>>> = Vec::new();
    for def in &pkg.graphics {
        let g = r.graphics(def);
        let at = def.id as usize;
        if graphics.len() <= at {
            graphics.resize(at + 1, None);
        }
        graphics[at] = g;
    }
    // Textures before the heaps: heap views are made on them.
    for def in &pkg.texture_defs {
        match draw::make_texture(m, def) {
            Ok(t) => {
                let t = Arc::new(t);
                m.textures.lock().unwrap_or_else(|e| e.into_inner()).insert(def.key as usize, Some(t.clone()));
                r.textures.insert(def.id, t);
                r.report.textures_made += 1;
            }
            Err(e) => r.report.texture_errors.push(format!("texture {}: {e}", def.id)),
        }
    }
    let mut texture_file = std::fs::File::open(pkg.dir.join("textures.bin")).ok();
    for def in &pkg.heaps {
        let h = m.new_heap(def.count, def.samplers).ok_or_else(|| format!("heap {}: could not be made", def.id))?;
        if h.capacity != def.capacity {
            r.report.heap_capacity_mismatches += 1;
        }
        r.heaps.insert(def.id, (h.buffer.clone(), h.strides.as_ref().map(|(b, _)| b.clone())));
        m.heaps.lock().unwrap_or_else(|e| e.into_inner()).insert(HEAP_KEY + def.id as usize, h);
    }
    let mut file = std::fs::File::open(pkg.dir.join("memory.bin")).map_err(|e| format!("memory.bin: {e}"))?;
    for (run, ops) in &pkg.runs {
        r.report.pages_written += write_epoch(pkg, &mut file, *run)? as u64;
        for v in pkg.texture_versions.iter().filter(|v| v.epoch == *run) {
            let (Some(f), Some(def), Some(t)) = (texture_file.as_mut(), pkg.texture_defs.get(v.id as usize), r.textures.get(&v.id)) else { continue };
            let bytes = read_blob(f, v.offset, v.len)?;
            if draw::write_texture(m, &t.texture, def, &bytes) {
                r.report.texture_uploads += 1;
            } else {
                r.report.texture_errors.push(format!("texture {}: upload failed", v.id));
            }
        }
        r.apply_updates(*run);
        if opts.runs.is_some_and(|(a, b)| *run < a || *run >= b) {
            continue;
        }
        let ops: Vec<Arc<MetalOp>> = ops.iter().filter_map(|op| r.op(op, &pipelines, &graphics)).map(Arc::new).collect();
        r.report.ops_run += ops.len() as u64;
        r.report.runs += 1;
        if !m.run(&ops) {
            r.report.failed_run = Some(*run);
            break;
        }
    }
    Ok(r.report)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Compare {
    pub(crate) pages: u64,
    pub(crate) identical: u64,
    /// (page, first differing byte, differing bytes).
    pub(crate) differing: Vec<(u64, usize, usize)>,
}

/// The pages the GPU wrote while recording, compared with memory now.
pub(crate) fn compare_reference(pkg: &Package) -> Result<Compare, String> {
    let mut file = std::fs::File::open(pkg.dir.join("reference").join("memory.bin")).map_err(|e| format!("reference/memory.bin: {e}"))?;
    let mut out = Compare::default();
    for m in &pkg.reference {
        let want = pkg.page_bytes(&mut file, m)?;
        // SAFETY: placed by `place`; the GPU is done (every run was waited for).
        let got = unsafe { std::slice::from_raw_parts(m.va as *const u8, want.len()) };
        out.pages += 1;
        let diff = want.iter().zip(got).filter(|(a, b)| a != b).count();
        if diff == 0 {
            out.identical += 1;
        } else {
            let first = want.iter().zip(got).position(|(a, b)| a != b).unwrap_or(0);
            out.differing.push((m.va, first, diff));
        }
    }
    Ok(out)
}

fn read_blob(file: &mut std::fs::File, offset: u64, len: u64) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0u8; len as usize];
    file.seek(SeekFrom::Start(offset)).and_then(|_| file.read_exact(&mut bytes)).map_err(|e| format!("blob at {offset:#x}: {e}"))?;
    Ok(bytes)
}

/// The textures the frame wrote, read back from the replay's own textures
/// and compared with the reference: (texture id, first differing byte,
/// differing bytes) in `differing`.
pub(crate) fn compare_textures(m: &MetalBackend, pkg: &Package) -> Result<Compare, String> {
    let mut file = std::fs::File::open(pkg.dir.join("reference").join("textures.bin")).map_err(|e| format!("reference/textures.bin: {e}"))?;
    let mut out = Compare::default();
    for v in &pkg.texture_reference {
        let want = read_blob(&mut file, v.offset, v.len)?;
        let def = pkg.texture_defs.get(v.id as usize).ok_or("reference names an unknown texture")?;
        let t = m.textures.lock().unwrap_or_else(|e| e.into_inner()).get(&(def.key as usize)).cloned().flatten();
        out.pages += 1;
        let Some(got) = t.and_then(|t| draw::read_texture(m, &t.texture, def)) else {
            out.differing.push((u64::from(v.id), 0, want.len()));
            continue;
        };
        let diff = want.iter().zip(&got).filter(|(a, b)| a != b).count() + want.len().abs_diff(got.len());
        if diff == 0 {
            out.identical += 1;
        } else {
            let first = want.iter().zip(&got).position(|(a, b)| a != b).unwrap_or(0);
            out.differing.push((u64::from(v.id), first, diff));
        }
    }
    Ok(out)
}
