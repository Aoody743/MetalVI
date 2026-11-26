//! MTLSamplerState wrapper: descriptor in Rust, built through the unix side's
//! `newSamplerStateWithDescriptor:`.

use crate::sys;
use crate::{Device, Error, Object, Pool, Result};
use sys::obj_handle_t;

/// A Metal sampler state (+1 reference released on drop; `Clone` retains).
#[derive(Clone, Debug)]
pub struct SamplerState(Object);

impl SamplerState {
    /// Its `gpuResourceID`, for descriptor entries.
    pub fn gpu_resource_id(&self) -> u64 {
        self.0.gpu_resource_id()
    }

    pub fn handle(&self) -> obj_handle_t {
        self.0.handle()
    }
}

/// Safe mirror of the bindgen `WMTSamplerInfo`.
#[derive(Clone, Debug)]
pub struct SamplerDescriptor {
    pub min_filter: sys::WMTSamplerMinMagFilter,
    pub mag_filter: sys::WMTSamplerMinMagFilter,
    pub mip_filter: sys::WMTSamplerMipFilter,
    pub r_address_mode: sys::WMTSamplerAddressMode,
    pub s_address_mode: sys::WMTSamplerAddressMode,
    pub t_address_mode: sys::WMTSamplerAddressMode,
    pub border_color: sys::WMTSamplerBorderColor,
    pub compare_function: sys::WMTCompareFunction,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub max_anisotropy: u32,
    pub normalized_coords: bool,
    pub lod_average: bool,
    pub support_argument_buffers: bool,
}

impl Default for SamplerDescriptor {
    fn default() -> Self {
        Self {
            min_filter: sys::WMTSamplerMinMagFilter::WMTSamplerMinMagFilterNearest,
            mag_filter: sys::WMTSamplerMinMagFilter::WMTSamplerMinMagFilterNearest,
            mip_filter: sys::WMTSamplerMipFilter::WMTSamplerMipFilterNotMipmapped,
            r_address_mode: sys::WMTSamplerAddressMode::WMTSamplerAddressModeClampToEdge,
            s_address_mode: sys::WMTSamplerAddressMode::WMTSamplerAddressModeClampToEdge,
            t_address_mode: sys::WMTSamplerAddressMode::WMTSamplerAddressModeClampToEdge,
            border_color: sys::WMTSamplerBorderColor::WMTSamplerBorderColorTransparentBlack,
            compare_function: sys::WMTCompareFunction::WMTCompareFunctionAlways,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1000.0,
            max_anisotropy: 1,
            normalized_coords: true,
            lod_average: false,
            support_argument_buffers: false,
        }
    }
}

impl Device {
    /// A sampler state from the descriptor (`newSamplerStateWithDescriptor:`).
    pub fn new_sampler_state(&self, desc: &SamplerDescriptor) -> Result<SamplerState> {
        let _pool = Pool::new();
        // SAFETY: all fields are plain values; `gpu_resource_id` is written
        // back by the unix side and starts at 0.
        let mut info = sys::WMTSamplerInfo {
            min_filter: desc.min_filter,
            mag_filter: desc.mag_filter,
            mip_filter: desc.mip_filter,
            r_address_mode: desc.r_address_mode,
            s_address_mode: desc.s_address_mode,
            t_address_mode: desc.t_address_mode,
            border_color: desc.border_color,
            compare_function: desc.compare_function,
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            max_anisotroy: desc.max_anisotropy,
            normalized_coords: desc.normalized_coords,
            lod_average: desc.lod_average,
            support_argument_buffers: desc.support_argument_buffers,
            gpu_resource_id: 0,
        };
        // SAFETY: `info` is valid for the call; `newSamplerState` returns owned (+1).
        let sampler = Object::adopt(unsafe { sys::MTLDevice_newSamplerState(self.handle(), &mut info) })
            .ok_or(Error("newSamplerState".into()))?;
        Ok(SamplerState(sampler))
    }
}
