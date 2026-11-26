//! Present path: a `CAMetalLayer` attached to a Win32 `HWND`, its drawables, and
//! the final `presentDrawable` call.
//!
//! This mirrors dxmt's swapchain path (see `metal-backend/flash/F18/NOTES.md`):
//! `CreateMetalViewFromHWND` creates a MetalView over the window's client view and
//! hands back the layer as a *weak* reference owned by that view; the layer lives
//! exactly as long as the view. Dropping the [`MetalLayer`] releases the view with
//! `ReleaseMetalView`.
//!
//! Every fallible entry returns [`crate::Error`]; nothing here panics on a null
//! Metal result.

use crate::render::{ColorAttachment, ColorDesc, Render, RenderPipeline, RenderPipelineDesc};
use crate::texture::Texture;
use crate::{CommandBuffer, Device, Error, Object, Pool, Result, error_text, sys};
use sys::obj_handle_t;

pub mod convert;
pub use convert::{PresentConverter, PresentOptions};

/// A MetalView over a Win32 window, exposing its `CAMetalLayer`.
///
/// The layer handle is borrowed from the view; do not use it after this value
/// drops.
pub struct MetalLayer {
    view: obj_handle_t,
    layer: obj_handle_t,
}

impl MetalLayer {
    /// Creates the MetalView over `hwnd` and returns the layer attached to it.
    ///
    /// The caller keeps the returned [`MetalLayer`] alive for as long as it draws
    /// to the window; dropping it releases the view and the layer with it.
    pub fn for_window(device: &Device, hwnd: isize) -> Result<Self> {
        let _pool = Pool::new();
        let mut layer: obj_handle_t = 0;
        // SAFETY: `device` is a live Metal device; `layer` is an out param the
        // callee fills for the duration of the call.
        let view = unsafe { sys::CreateMetalViewFromHWND(hwnd, device.handle(), &mut layer) };
        if view == 0 || layer == 0 {
            return Err(Error(format!(
                "CreateMetalViewFromHWND(hwnd={hwnd:#x}) failed (view={view:#x}, layer={layer:#x})"
            )));
        }
        Ok(Self { view, layer })
    }

    /// The raw layer handle (for sys-level calls).
    pub fn handle(&self) -> obj_handle_t {
        self.layer
    }

    /// Reads the layer's current properties back from Metal.
    pub fn properties(&self) -> Result<sys::WMTLayerProps> {
        let _pool = Pool::new();
        let mut props: sys::WMTLayerProps = unsafe { std::mem::zeroed() };
        // SAFETY: `props` is writable for the call; the layer is live.
        unsafe { sys::MetalLayer_getProps(self.layer, &mut props) };
        Ok(props)
    }

    /// The current `drawableSize` as `(width, height)`.
    pub fn drawable_size(&self) -> Result<(f64, f64)> {
        let props = self.properties()?;
        Ok((props.drawable_width, props.drawable_height))
    }

    /// Updates only the drawable size, leaving the other layer properties
    /// (device, opaque, display sync, framebuffer-only, pixel format, contents
    /// scale) untouched: reads the current props, overrides the size, pushes
    /// them back through `setProps`.
    pub fn set_drawable_size(&self, width: f64, height: f64) -> Result<()> {
        let mut props = self.properties()?;
        props.drawable_width = width;
        props.drawable_height = height;
        self.set_props(&props);
        Ok(())
    }

    /// Pushes a full property set to the layer (`setProps`).
    pub fn set_props(&self, props: &sys::WMTLayerProps) {
        // SAFETY: the layer is live; `props` outlives the call.
        unsafe { sys::MetalLayer_setProps(self.layer, props) };
    }

    /// Sets the layer's colorspace (and HDR content flag on the unix side).
    /// Fails if the unix side reports the colorspace could not be applied.
    pub fn set_color_space(&self, color_space: sys::WMTColorSpace) -> Result<()> {
        // SAFETY: the layer is live.
        let ok = unsafe { sys::MetalLayer_setColorSpace(self.layer, color_space) };
        if !ok {
            return Err(Error(format!(
                "MetalLayer_setColorSpace({}) refused",
                color_space.0
            )));
        }
        Ok(())
    }

    /// Returns the next drawable for the layer. The drawable is retained on
    /// return and released when the [`Drawable`] drops; pair it with a command
    /// buffer's [`present`](crate::CommandBuffer::present).
    pub fn next_drawable(&self) -> Result<Drawable> {
        let _pool = Pool::new();
        // SAFETY: the layer is live.
        let drawable = unsafe { sys::MetalLayer_nextDrawable(self.layer) };
        if drawable == 0 {
            return Err(Error("MetalLayer_nextDrawable returned null".into()));
        }
        // SAFETY: the nextDrawable result is an ObjC object; retain it so the
        // inner autorelease pool draining above cannot deallocate it before the
        // frame finishes.
        unsafe { sys::NSObject_retain(drawable) };
        Ok(Drawable(drawable))
    }
}

impl Drop for MetalLayer {
    fn drop(&mut self) {
        if self.view != 0 {
            // SAFETY: the view handle came from CreateMetalViewFromHWND and has
            // not been released yet.
            unsafe { sys::ReleaseMetalView(self.view) };
            self.view = 0;
        }
    }
}

/// A drawable handed out by `nextDrawable`; its texture is the render target of
/// the frame's render pass.
pub struct Drawable(obj_handle_t);

impl Drawable {
    /// The raw drawable handle (for sys-level calls).
    pub fn handle(&self) -> obj_handle_t {
        self.0
    }

    /// The drawable's texture, retained so it outlives the drawable borrow.
    /// Use its handle with `MTLTexture_*` queries (size, pixel format).
    pub fn texture(&self) -> Result<Object> {
        let _pool = Pool::new();
        // SAFETY: the drawable is live.
        let texture = unsafe { sys::MetalDrawable_texture(self.0) };
        if texture == 0 {
            return Err(Error("MetalDrawable_texture returned null".into()));
        }
        // SAFETY: a live texture handle from the unix side; retain so the owned
        // Object's release balances it after the drawable is gone.
        Object::retain(texture).ok_or_else(|| Error("drawable texture retain failed".into()))
    }
}

impl Drop for Drawable {
    fn drop(&mut self) {
        if self.0 != 0 {
            // SAFETY: the handle came from nextDrawable and was retained once.
            unsafe { sys::NSObject_release(self.0) };
            self.0 = 0;
        }
    }
}

/// A single-plane SDR copy from a rendered texture to a BGRA8 drawable.
/// Fits the source without enlargement and clears the surrounding area to black.
pub struct PresentBlit(RenderPipeline);

impl PresentBlit {
    pub fn new(device: &Device) -> Result<Self> {
        let _pool = Pool::new();
        let source = include_str!("present.metal");
        let mut err = 0;
        // SAFETY: source and the error out-parameter are live for the call.
        let lib = Object::adopt(unsafe {
            sys::MTLDevice_newLibraryWithSource(device.handle(), source.as_ptr().cast(), source.len() as u64, &mut err)
        })
        .ok_or_else(|| error_text(err, "present blit shader"))?;
        let mut desc = RenderPipelineDesc { topology_class: 3, sample_count: 1, ..Default::default() };
        desc.colors[0] = ColorDesc { pixel_format: sys::WMTPixelFormat::WMTPixelFormatBGRA8Unorm.0, write_mask: 15, ..Default::default() };
        device.render_pipeline_desc((&lib, "present_vertex"), Some((&lib, "present_fragment")), &desc).map(Self)
    }

    /// Encodes to a drawable and schedules it for presentation on `cb`.
    pub fn draw_to_window(&self, cb: &CommandBuffer, src: &Texture, drawable: &Drawable) -> Result<()> {
        self.draw(cb, src, &Texture::from_object(drawable.texture()?))?;
        cb.present(drawable);
        Ok(())
    }

    /// Encodes the pass without presenting, also allowing offscreen readback.
    /// Both resources must be distinct, single-sampled 2D textures on the
    /// same device, with shader-read/render-target usage respectively.
    pub fn draw(&self, cb: &CommandBuffer, src: &Texture, dst: &Texture) -> Result<()> {
        use sys::WMTPixelFormat as Format;
        if src.handle() == dst.handle()
            || !matches!(src.pixel_format(), Format::WMTPixelFormatRGBA8Unorm | Format::WMTPixelFormatBGRA8Unorm)
            || dst.pixel_format() != Format::WMTPixelFormatBGRA8Unorm
        {
            return Err(Error("present blit needs a distinct RGBA8/BGRA8 UNorm source and BGRA8 UNorm target".into()));
        }
        let [sw, sh] = [src.width() as u32, src.height() as u32];
        let [dw, dh] = [dst.width() as u32, dst.height() as u32];
        if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
            return Err(Error("empty present texture".into()));
        }
        let scale = 1.0f64.min(f64::from(dw) / f64::from(sw)).min(f64::from(dh) / f64::from(sh));
        let [w, h] = [sw, sh].map(|dim| ((f64::from(dim) * scale).round() as u32).max(1));
        let [x, y] = [(dw - w) / 2, (dh - h) / 2];
        let constants = [x, y, w, h].map(u32::to_le_bytes);
        let viewport = sys::WMTViewport {
            originX: x.into(),
            originY: y.into(),
            width: w.into(),
            height: h.into(),
            znear: 0.0,
            zfar: 1.0,
        };
        cb.render_pass(
            &[ColorAttachment {
                texture: dst,
                load_action: sys::WMTLoadAction::WMTLoadActionClear,
                store_action: sys::WMTStoreAction::WMTStoreActionStore,
                clear: [0., 0., 0., 1.],
            }],
            dw,
            dh,
            &[
                Render::Pipeline { pipeline: &self.0 },
                Render::Viewports { viewports: std::slice::from_ref(&viewport) },
                Render::FragmentTexture { texture: src, index: 0 },
                Render::FragmentBytes { bytes: constants.as_flattened(), index: 0 },
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

impl crate::CommandBuffer {
    /// Presents `drawable` on this command buffer (`presentDrawable`).
    ///
    /// The drawable must be the one returned by the layer this buffer draws to;
    /// call this after the frame's render pass has been encoded.
    pub fn present(&self, drawable: &Drawable) {
        // SAFETY: both the command buffer and the drawable are live.
        unsafe { sys::MTLCommandBuffer_presentDrawable(self.handle(), drawable.handle()) };
    }

    /// Presents `drawable` with a minimum display duration in seconds
    /// (`presentDrawableAfterMinimumDuration`), the frame-rate throttle dxmt
    /// uses instead of `displaySyncEnabled`.
    pub fn present_after_minimum_duration(&self, drawable: &Drawable, after: f64) {
        // SAFETY: both are live; `after` is the minimum duration in seconds.
        unsafe {
            sys::MTLCommandBuffer_presentDrawableAfterMinimumDuration(
                self.handle(),
                drawable.handle(),
                after,
            )
        };
    }
}
