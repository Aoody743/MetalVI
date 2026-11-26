//! End-to-end binding test (run in Wine with victorymetal installed): a
//! known SM 6.6 compute shader (metal-backend/tools/dxc/test/bind.hlsl,
//! compiled by DXC) through Metal Shader Converter, bound the way the Metal
//! backend binds title shaders:
//! - root constants at b0 space0 (victory's user data) in the top-level
//!   argument buffer (bind point 2);
//! - a root UAV at u0 space1 (victory's root views): a GPU address in it;
//! - an input buffer through the directly indexed descriptor heap (bind
//!   point 0), slot 5.
//!
//! usage: victory-metal-bind <bind.dxil>

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => {
            println!("RESULT: ALL CHECKS PASSED");
            std::process::ExitCode::SUCCESS
        }
        Err(e) => {
            println!("RESULT: FAILED: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use victory_metal::ir::{DescriptorEntry, ResourceType};
    use victory_metal::msc::{self, RootParam};
    use victory_metal::{Compute, Device, Pool, ResidencySet};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};

    const COUNT: u32 = 10_000;
    const SLOT: u32 = 5;
    const ADD: u32 = 100;
    let path = std::env::args().nth(1).ok_or("usage: victory-metal-bind <bind.dxil>")?;
    let dxil = std::fs::read(&path)?;
    let _pool = Pool::new();
    let device = Device::first()?;

    let alloc = |bytes: usize| {
        // SAFETY: a fresh allocation.
        let p = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        if p.is_null() { Err("VirtualAlloc") } else { Ok(p) }
    };
    let bytes = 64 << 10;
    let (input_mem, output_mem, heap_mem) = (alloc(bytes)?, alloc(bytes)?, alloc(bytes)?);
    // SAFETY: committed above, COUNT u32s fit.
    let input = unsafe { std::slice::from_raw_parts_mut(input_mem.cast::<u32>(), COUNT as usize) };
    let output = unsafe { std::slice::from_raw_parts_mut(output_mem.cast::<u32>(), COUNT as usize) };
    for (i, v) in input.iter_mut().enumerate() {
        *v = i as u32 ^ 0x5a5a;
    }
    output.fill(0xCCCC_CCCC);

    // SAFETY: the allocations live until the end of run(), after the GPU is done.
    let (input_buf, output_buf, heap_buf) =
        unsafe { (device.wrap(input_mem, bytes as u64)?, device.wrap(output_mem, bytes as u64)?, device.wrap(heap_mem, bytes as u64)?) };

    // The descriptor heap: slot SLOT is the input buffer, the rest empty.
    // SAFETY: the heap memory holds bytes/24 entries.
    let heap = unsafe { std::slice::from_raw_parts_mut(heap_mem.cast::<DescriptorEntry>(), bytes / 24) };
    heap.fill(DescriptorEntry::NULL);
    heap[SLOT as usize] = DescriptorEntry::buffer(input_buf.gpu_address, u64::from(COUNT) * 4);

    // The root signature victory would build for such a shader.
    let params = [RootParam::Constants { register: 0, space: 0, num: 4 }, RootParam::Uav { register: 0, space: 1 }];
    let (pso, mut converted) = device.compute_pipeline_dxil(&dxil, &params, &[])?;
    println!(
        "pipeline: entry {:?} threadgroup {:?} top-level {:?}",
        converted.entry,
        converted.threadgroup,
        converted.top_level.layout.iter().map(|l| (l.kind, l.space, l.slot, l.offset, l.size)).collect::<Vec<_>>()
    );
    let top = &mut converted.top_level;
    if !top.set_constants(0, 0, &[SLOT, ADD, COUNT, 0]) {
        return Err("no root constants at b0 space0 in the layout".into());
    }
    if !top.set_root_descriptor(ResourceType::Uav, 1, 0, output_buf.gpu_address) {
        return Err("no root UAV at u0 space1 in the layout".into());
    }

    // Shaders reach the input and output only by address: make them resident.
    let residency = ResidencySet::new(&device, 8)?;
    residency.add(&[input_buf.handle(), output_buf.handle(), heap_buf.handle()]);
    residency.commit();
    let queue = device.new_queue(4)?;
    queue.add_residency_set(&residency);

    let tg = converted.threadgroup.map(u64::from);
    let groups = u64::from(COUNT).div_ceil(tg[0].max(1));
    let cb = queue.command_buffer()?;
    cb.compute(&[
        Compute::Pipeline { pipeline: &pso, threads_per_group: tg },
        Compute::Buffer { buffer: &heap_buf, offset: 0, index: msc::DESCRIPTOR_HEAP_BIND_POINT },
        Compute::Bytes { bytes: &converted.top_level.bytes, index: msc::ARGUMENT_BUFFER_BIND_POINT },
        Compute::Dispatch { groups: [groups, 1, 1] },
    ])?;
    cb.commit();
    cb.wait()?;

    let bad: Vec<usize> = (0..COUNT as usize).filter(|&i| output[i] != input[i].wrapping_mul(3).wrapping_add(ADD)).collect();
    println!("{COUNT} outputs: {} wrong", bad.len());
    if let Some(&i) = bad.first() {
        println!("first wrong: [{i}] = {:#x}, want {:#x}", output[i], input[i].wrapping_mul(3).wrapping_add(ADD));
        return Err("wrong results".into());
    }
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-bind runs on Windows (Wine) only");
}
