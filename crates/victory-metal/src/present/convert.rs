use super::{Drawable, MetalLayer};
use crate::render::{ColorAttachment, ColorDesc, Render, RenderPipeline, RenderPipelineDesc};
use crate::texture::Texture;
use crate::{CommandBuffer, Device, Error, Object, Pool, Result, sys};
use sys::WMTPixelFormat as Format;

/// Options for a single-plane SDR present.
#[derive(Clone, Copy, Debug)]
pub struct PresentOptions {
    /// Treat RGB10A2 or 8-bit UNorm as linear rather than sRGB-encoded SDR.
    /// This does not override the transfer of a typed sRGB or float input.
    pub unorm_is_linear: bool,
    /// Luminance at the SDR shoulder knee for scRGB (whose unit is 80 nits).
    pub scrgb_paper_white_nits: f32,
}

impl Default for PresentOptions {
    fn default() -> Self {
        Self {
            unorm_is_linear: false,
            scrgb_paper_white_nits: 80.0,
        }
    }
}

/// Converts SDR or scRGB source textures into an SDR sRGB drawable.
pub struct PresentConverter {
    device: Device,
    library: Object,
    pipelines: [Option<RenderPipeline>; 2],
}

impl PresentConverter {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            library: device.library_msl(include_str!("convert.metal"))?,
            pipelines: [None, None],
        })
    }

    /// The default window draws sRGB code values into BGRA8Unorm. Setting the
    /// sRGB layer color space also disables the layer's EDR flag in the shim.
    /// Configure the layer before requesting its first drawable.
    pub fn configure_sdr_layer(layer: &MetalLayer) -> Result<()> {
        let _pool = Pool::new();
        let mut props = layer.properties()?;
        layer.set_color_space(sys::WMTColorSpace::WMTColorSpaceSRGB)?;
        props.pixel_format = Format::WMTPixelFormatBGRA8Unorm;
        props.opaque = true;
        layer.set_props(&props);
        Ok(())
    }

    pub fn draw_to_window(
        &mut self,
        cb: &CommandBuffer,
        src: &Texture,
        source_dxgi: u32,
        drawable: &Drawable,
        options: PresentOptions,
    ) -> Result<()> {
        self.draw(
            cb,
            src,
            source_dxgi,
            &Texture::from_object(drawable.texture()?),
            options,
        )?;
        cb.present(drawable);
        Ok(())
    }

    /// Encodes a centered aspect-fit pass, including enlargement, with opaque
    /// black bars. Both textures must be distinct, single-sample 2D textures
    /// on this device with shader-read and render-target usage respectively.
    /// `source_dxgi` is the typed format, not a typeless resource family.
    pub fn draw(
        &mut self,
        cb: &CommandBuffer,
        src: &Texture,
        source_dxgi: u32,
        dst: &Texture,
        options: PresentOptions,
    ) -> Result<()> {
        let (src_format, mode) = source_mode(source_dxgi, options)?;
        if src.handle() == dst.handle() || src.pixel_format() != src_format {
            return Err(Error(
                "present converter needs distinct textures and a matching DXGI/Metal source format"
                    .into(),
            ));
        }
        let format = dst.pixel_format();
        let target = match format {
            Format::WMTPixelFormatBGRA8Unorm => 0,
            Format::WMTPixelFormatBGRA8Unorm_sRGB => 1,
            _ => {
                return Err(Error(format!(
                    "unsupported SDR drawable format {}",
                    format.0
                )));
            }
        };
        let exposure = if mode == 2 {
            let exposure = 80.0 / options.scrgb_paper_white_nits;
            if !exposure.is_finite() || exposure <= 0.0 {
                return Err(Error("invalid scRGB paper white".into()));
            }
            exposure
        } else {
            1.0
        };
        let [sw, sh, dw, dh] = [src.width(), src.height(), dst.width(), dst.height()];
        if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
            return Err(Error("empty present texture".into()));
        }
        let [sw, sh, tw, th] = [sw, sh, dw, dh].map(|dim| dim as f64);
        let scale = (tw / sw).min(th / sh);
        let [vw, vh] = [sw * scale, sh * scale];
        let viewport = sys::WMTViewport {
            originX: (tw - vw) / 2.0,
            originY: (th - vh) / 2.0,
            width: vw,
            height: vh,
            znear: 0.0,
            zfar: 1.0,
        };
        let constants = [mode, target as u32, exposure.to_bits()].map(u32::to_le_bytes);
        if self.pipelines[target].is_none() {
            let mut desc = RenderPipelineDesc {
                topology_class: 3,
                sample_count: 1,
                ..Default::default()
            };
            desc.colors[0] = ColorDesc {
                pixel_format: format.0,
                write_mask: 15,
                ..Default::default()
            };
            self.pipelines[target] = Some(self.device.render_pipeline_desc(
                (&self.library, "convert_vertex"),
                Some((&self.library, "convert_fragment")),
                &desc,
            )?);
        }
        let pipeline = self.pipelines[target].as_ref().expect("initialized above");
        cb.render_pass(
            &[ColorAttachment {
                texture: dst,
                load_action: sys::WMTLoadAction::WMTLoadActionClear,
                store_action: sys::WMTStoreAction::WMTStoreActionStore,
                clear: [0., 0., 0., 1.],
            }],
            dw as u32,
            dh as u32,
            &[
                Render::Pipeline { pipeline },
                Render::Viewports {
                    viewports: std::slice::from_ref(&viewport),
                },
                Render::FragmentTexture {
                    texture: src,
                    index: 0,
                },
                Render::FragmentBytes {
                    bytes: constants.as_flattened(),
                    index: 0,
                },
                Render::Draw {
                    primitive_type: sys::WMTPrimitiveType::WMTPrimitiveTypeTriangle,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
            ],
        )
    }
}

fn source_mode(dxgi: u32, options: PresentOptions) -> Result<(Format, u32)> {
    let unorm = u32::from(options.unorm_is_linear);
    Ok(match dxgi {
        10 => (Format::WMTPixelFormatRGBA16Float, 2), // R16G16B16A16_FLOAT
        24 => (Format::WMTPixelFormatRGB10A2Unorm, unorm), // R10G10B10A2_UNORM
        28 => (Format::WMTPixelFormatRGBA8Unorm, unorm), // R8G8B8A8_UNORM
        29 => (Format::WMTPixelFormatRGBA8Unorm_sRGB, 1), // R8G8B8A8_UNORM_SRGB
        87 | 88 => (Format::WMTPixelFormatBGRA8Unorm, unorm), // B8G8R8A8/X8_UNORM
        91 | 93 => (Format::WMTPixelFormatBGRA8Unorm_sRGB, 1), // B8G8R8A8/X8_UNORM_SRGB
        _ => return Err(Error(format!("unsupported DXGI present format {dxgi}"))),
    })
}

#[cfg(test)]
mod tests;
