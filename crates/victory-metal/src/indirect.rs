//! GPU argument translation for the four single-operation ExecuteIndirect signatures.

use crate::msc::{DRAW_ARGUMENTS_BIND_POINT, DRAW_UNIFORMS_BIND_POINT};
use crate::render::Render;
use crate::{Buffer, CommandBuffer, Compute, ComputePipeline, Device, Error, Result};

const SOURCE: &str = include_str!("indirect.metal");
const RECORD_BYTES: u64 = 20;
const TRACE_BYTES: usize = 48;
// Both Metal and the MSC draw uniform accept 4-byte offsets on Apple silicon.
const DRAW_PARAMS_BYTES: u64 = 20;
const THREADS: u64 = 64;
pub const RECORD_LIMIT: u32 = 64;
pub const GROUPS_PER_RECORD: u32 = 32;
pub const INPUTS_PER_RECORD: u32 = 512;
/// Draws a bounded rectangle record is cut into, each of at most the group
/// budget (`record_budget().0` object groups): with 32 groups of 255
/// vertices, 16 draws cover 130,560 vertices (43,520 rectangles, a 4K
/// frame of 16x16 tiles). Before, a record was cut to its first 510 vertices
/// (the input budget): tiled lighting and post passes drew only their first
/// tiles, the regular blue blocks of the window. The rest past the segments
/// is still clamped and counted.
pub const RECT_SEGMENTS: u32 = 16;

/// The per-record budget the clamp applies: `GROUPS_PER_RECORD` and
/// `INPUTS_PER_RECORD` unless `VICTORY_METAL_INDIRECT_GROUPS` /
/// `VICTORY_METAL_INDIRECT_INPUTS` give larger finite values (at most 2^20
/// groups and 2^26 inputs, the ABI bounds the kernel applies anyway). The
/// clamp stays on either way; the fuse bounds the measured GPU time.
///
/// At 32 groups every indirect dispatch and rectangle draw of the 3D frames
/// was cut short (~15,000 clamp lines a run from frame ~1040): tiles of
/// the lighting and post passes left undrawn, the blue blocks of the window.
pub fn record_budget() -> (u32, u32) {
    static BUDGET: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        let var = |name: &str, default: u32, max: u32| {
            std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok()).map_or(default, |v| v.clamp(default, max))
        };
        (var("VICTORY_METAL_INDIRECT_GROUPS", GROUPS_PER_RECORD, 1 << 20),
         var("VICTORY_METAL_INDIRECT_INPUTS", INPUTS_PER_RECORD, 1 << 26))
    })
}

pub struct ClampCounter(Buffer, *mut u8);

impl ClampCounter {
    pub fn new(device: &Device) -> Result<Self> {
        let (b, ptr) = device.shared_buffer_mapped(&0u32.to_ne_bytes())?;
        Ok(Self(b, ptr))
    }

    /// # Safety
    /// All commands using this counter must have completed before reading.
    pub unsafe fn completed_count(&self) -> u32 {
        unsafe { self.1.cast::<u32>().read_volatile() }
    }
}

pub struct IndirectTrace {
    buffer: Buffer,
    ptr: *mut u8,
    count: u32,
}

pub struct IndirectTraceRecord {
    pub raw_count: u32,
    pub active: bool,
    pub before: [u32; 5],
    pub after: [u32; 5],
}

impl IndirectTrace {
    /// # Safety
    /// The command buffer that wrote this trace must have completed successfully.
    pub unsafe fn completed_records(&self) -> impl Iterator<Item = IndirectTraceRecord> + '_ {
        let bytes = unsafe { std::slice::from_raw_parts(self.ptr, self.count as usize * TRACE_BYTES) };
        bytes.chunks_exact(TRACE_BYTES).map(|bytes| {
            let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            IndirectTraceRecord {
                raw_count: word(0),
                active: word(1) != 0,
                before: std::array::from_fn(|i| word(i + 2)),
                after: std::array::from_fn(|i| word(i + 7)),
            }
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum IndirectKind {
    Draw,
    /// Capacity of the logical index view in elements, excluding window padding.
    /// Metal index types are 0 for u16 and 1 for u32.
    DrawIndexed {
        index_capacity: u32,
        index_type: u8,
    },
    Dispatch,
    DrawMesh,
    TitleMesh1D,
    /// Non-indexed three-vertex rectangles expanded by MSC's object/mesh pair.
    Rectangles(crate::msc::GeometryConfig),
}

pub struct IndirectSource<'a> {
    pub kind: IndirectKind,
    pub records: (&'a Buffer, u64),
    pub stride: u32,
    pub max_count: u32,
    pub count: Option<(&'a Buffer, u64)>,
}

pub struct IndirectTranslator(ComputePipeline);

impl IndirectTranslator {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self(
            device.compute_pipeline_msl(SOURCE, "translate_indirect")?,
        ))
    }

    /// Encodes the translation immediately before the consuming encoder.
    /// A GPU producer of records/count must precede this pass. The encoder
    /// boundary orders tracked buffers; GPU-address loads/stores also need
    /// the caller's usual resource declarations and synchronization.
    ///
    /// This handles the argument ABI, not the title's list-wide work budget
    /// or the special dispatch splitting that rewrites a root constant.
    pub fn encode(
        &self,
        device: &Device,
        command: &CommandBuffer,
        source: &IndirectSource<'_>,
    ) -> Result<Option<IndirectBatch>> {
        self.encode_impl(device, command, source, None, false)
    }

    pub fn encode_bounded(&self, device: &Device, command: &CommandBuffer, source: &IndirectSource<'_>, counter: &ClampCounter) -> Result<Option<IndirectBatch>> {
        self.encode_bounded_with_trace(device, command, source, counter, false)
    }

    pub fn encode_bounded_with_trace(&self, device: &Device, command: &CommandBuffer, source: &IndirectSource<'_>, counter: &ClampCounter, trace: bool) -> Result<Option<IndirectBatch>> {
        if source.max_count > RECORD_LIMIT {
            return Err(Error("indirect record budget exceeded".into()));
        }
        self.encode_impl(device, command, source, Some(counter), trace)
    }

    fn encode_impl(&self, device: &Device, command: &CommandBuffer, source: &IndirectSource<'_>, counter: Option<&ClampCounter>, trace: bool) -> Result<Option<IndirectBatch>> {
        if source.max_count == 0 {
            return Ok(None);
        }
        let (kind, words, index_limit) = match source.kind {
            IndirectKind::Draw => (0, 4, 0),
            IndirectKind::DrawIndexed {
                index_capacity,
                index_type: 0..=1,
            } => (1, 5, index_capacity),
            IndirectKind::DrawIndexed { .. } => {
                return Err(Error("invalid indirect index type".into()));
            }
            IndirectKind::Dispatch => (2, 3, 0),
            IndirectKind::DrawMesh => (10, 3, 0),
            IndirectKind::TitleMesh1D => (12, 3, 0),
            IndirectKind::Rectangles(_) => (11, 4, 0),
        };
        let stride = u64::from(source.stride);
        let (records, records_offset) = source.records;
        let records_bytes = u64::from(source.max_count - 1) * stride + words * 4;
        if stride < words * 4
            || stride % 4 != 0
            || !valid_window(records, records_offset, records_bytes)
        {
            return Err(Error("invalid indirect argument window or stride".into()));
        }
        let (count, count_offset) = source.count.unwrap_or((records, 0));
        if source.count.is_some() && !valid_window(count, count_offset, 4) {
            return Err(Error("invalid indirect count window".into()));
        }
        // Output records per source record (see `RECT_SEGMENTS`).
        let segments = if matches!(source.kind, IndirectKind::Rectangles(_)) && counter.is_some() { RECT_SEGMENTS } else { 1 };
        let outputs = u64::from(source.max_count) * u64::from(segments);
        let rect = if let IndirectKind::Rectangles(cfg) = source.kind {
            Some((cfg.object_vertex_stride().ok_or_else(|| Error("invalid MSC geometry stride".into()))?, cfg.max_input_primitives))
        } else { None };
        let values = [
            kind,
            source.stride / 4,
            source.max_count,
            u32::from(source.count.is_some()),
            index_limit,
            rect.map_or(0, |r| r.0),
            rect.map_or(0, |r| r.1),
            if counter.is_some() { record_budget().0 } else { 0 },
            record_budget().1,
            u32::from(trace),
            segments,
        ];
        let mut options = [0; 44];
        for (dst, value) in options.chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        let output = device.private_buffer(outputs * RECORD_BYTES)?;
        let draw_params = (kind <= 1 || rect.is_some())
            .then(|| device.private_buffer(outputs * if rect.is_some() { 48 } else { DRAW_PARAMS_BYTES }))
            .transpose()?;
        let trace = if trace {
            let (buffer, ptr) = device.shared_buffer_mapped(&vec![0; source.max_count as usize * TRACE_BYTES])?;
            Some(IndirectTrace { buffer, ptr, count: source.max_count })
        } else { None };
        command.compute(&[
            Compute::Pipeline {
                pipeline: &self.0,
                threads_per_group: [THREADS, 1, 1],
            },
            Compute::Buffer {
                buffer: records,
                offset: records_offset,
                index: 0,
            },
            Compute::Buffer {
                buffer: count,
                offset: count_offset,
                index: 1,
            },
            Compute::Buffer {
                buffer: &output,
                offset: 0,
                index: 2,
            },
            Compute::Bytes {
                bytes: &options,
                index: 3,
            },
            Compute::Buffer {
                // The kernel does not access this slot for dispatch or mesh.
                buffer: draw_params.as_ref().unwrap_or(&output),
                offset: 0,
                index: 4,
            },
            Compute::Buffer { buffer: counter.map_or(&output, |c| &c.0), offset: 0, index: 5 },
            Compute::Buffer { buffer: trace.as_ref().map_or(&output, |t| &t.buffer), offset: 0, index: 6 },
            Compute::Dispatch {
                groups: [outputs.div_ceil(THREADS), 1, 1],
            },
        ])?;
        Ok(Some(IndirectBatch {
            output,
            draw_params,
            kind: source.kind,
            count: source.max_count * segments,
            trace,
        }))
    }
}

#[cfg(test)]
mod tests;

fn valid_window(buffer: &Buffer, offset: u64, bytes: u64) -> bool {
    offset % 4 == 0
        && offset
            .checked_add(bytes)
            .is_some_and(|end| end <= buffer.len)
}

/// Scratch arguments. Retain this while building and encoding the consumers.
pub struct IndirectBatch {
    output: Buffer,
    draw_params: Option<Buffer>,
    kind: IndirectKind,
    count: u32,
    trace: Option<IndirectTrace>,
}

impl IndirectBatch {
    pub fn take_trace(&mut self) -> Option<IndirectTrace> {
        self.trace.take()
    }
    /// The GPU writes exactly this many records, including inactive records
    /// (`RECT_SEGMENTS` per bounded rectangle record).
    pub fn record_count(&self) -> u32 {
        self.count
    }

    /// Use after the translation encoder; bounds are checked against max_count.
    pub fn record(&self, index: u32) -> Option<IndirectRecord<'_>> {
        (index < self.count).then(|| IndirectRecord {
            buffer: &self.output,
            offset: u64::from(index) * RECORD_BYTES,
            draw_params: self
                .draw_params
                .as_ref()
                .map(|buffer| (buffer, u64::from(index) * if matches!(self.kind, IndirectKind::Rectangles(_)) {48} else {DRAW_PARAMS_BYTES})),
            kind: self.kind,
        })
    }
}

pub struct IndirectRecord<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    draw_params: Option<(&'a Buffer, u64)>,
    kind: IndirectKind,
}

impl<'a> IndirectRecord<'a> {
    pub fn rectangle_bindings(&self) -> Option<[Render<'a>; 4]> {
        let IndirectKind::Rectangles(_) = self.kind else { return None };
        let (b, at) = self.draw_params?;
        Some([
            Render::ObjectBuffer { buffer: b, offset: at, index: DRAW_ARGUMENTS_BIND_POINT },
            Render::MeshBuffer { buffer: b, offset: at, index: DRAW_ARGUMENTS_BIND_POINT },
            Render::ObjectBuffer { buffer: b, offset: at + 24, index: DRAW_UNIFORMS_BIND_POINT },
            Render::MeshBuffer { buffer: b, offset: at + 24, index: DRAW_UNIFORMS_BIND_POINT },
        ])
    }
    /// Apply these just before this record's draw. Real mesh and compute
    /// functions do not use the vertex draw-parameter ABI.
    pub fn msc_vertex_bindings(&self) -> Option<[Render<'a>; 2]> {
        let (buffer, offset) = self.draw_params?;
        let mode = match self.kind {
            IndirectKind::Draw => &[0, 0],
            IndirectKind::DrawIndexed { index_type: 0, .. } => &[1, 0],
            IndirectKind::DrawIndexed { index_type: 1, .. } => &[2, 0],
            _ => return None,
        };
        Some([
            Render::VertexBuffer {
                buffer,
                offset,
                index: DRAW_ARGUMENTS_BIND_POINT,
            },
            Render::VertexBytes {
                bytes: mode,
                index: DRAW_UNIFORMS_BIND_POINT,
            },
        ])
    }
}
