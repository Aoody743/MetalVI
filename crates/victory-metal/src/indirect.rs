//! GPU argument translation for the four single-operation ExecuteIndirect signatures.

use crate::msc::{DRAW_ARGUMENTS_BIND_POINT, DRAW_UNIFORMS_BIND_POINT};
use crate::render::Render;
use crate::{Buffer, CommandBuffer, Compute, ComputePipeline, Device, Error, Result};

const SOURCE: &str = include_str!("indirect.metal");
const RECORD_BYTES: u64 = 20;
// Both Metal and the MSC draw uniform accept 4-byte offsets on Apple silicon.
const DRAW_PARAMS_BYTES: u64 = 20;
const THREADS: u64 = 64;

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
        let values: [u32; 5] = [
            kind,
            source.stride / 4,
            source.max_count,
            u32::from(source.count.is_some()),
            index_limit,
        ];
        let mut options = [0; 20];
        for (dst, value) in options.chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        let output = device.private_buffer(u64::from(source.max_count) * RECORD_BYTES)?;
        let draw_params = (kind <= 1)
            .then(|| device.private_buffer(u64::from(source.max_count) * DRAW_PARAMS_BYTES))
            .transpose()?;
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
            Compute::Dispatch {
                groups: [u64::from(source.max_count).div_ceil(THREADS), 1, 1],
            },
        ])?;
        Ok(Some(IndirectBatch {
            output,
            draw_params,
            kind: source.kind,
            count: source.max_count,
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
}

impl IndirectBatch {
    /// The GPU writes exactly this many records, including inactive records.
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
                .map(|buffer| (buffer, u64::from(index) * DRAW_PARAMS_BYTES)),
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
