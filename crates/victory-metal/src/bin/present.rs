//! Present-path smoke test (run in Wine with victorymetal installed):
//! create an overlapped Win32 window, attach a MetalLayer through
//! `CreateMetalViewFromHWND`, size the layer to the client area, then run 120
//! frames: a clear-only render pass over each drawable's texture with a
//! per-frame changing clear color, `presentDrawable`, commit and wait.
//!
//! user32 is resolved at runtime through `LoadLibraryA`/`GetProcAddress`; the
//! binary declares no extra windows-sys features and links no user32 import
//! library.

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
mod win32 {
    //! Minimal self-declared user32 FFI, resolved at runtime.

    use core::ffi::c_void;

    pub type HWND = *mut c_void;
    pub type HINSTANCE = *mut c_void;
    pub type HBRUSH = *mut c_void;
    pub type WPARAM = usize;
    pub type LPARAM = isize;
    pub type LRESULT = isize;

    #[repr(C)]
    pub struct WNDCLASSW {
        pub style: u32,
        pub lpfn_wnd_proc: Option<extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT>,
        pub cb_cls_extra: i32,
        pub cb_wnd_extra: i32,
        pub h_instance: HINSTANCE,
        pub h_icon: *mut c_void,
        pub h_cursor: *mut c_void,
        pub hbr_background: HBRUSH,
        pub lpsz_menu_name: *const u16,
        pub lpsz_class_name: *const u16,
    }

    #[repr(C)]
    pub struct RECT {
        pub left: i32,
        pub top: i32,
        pub right: i32,
        pub bottom: i32,
    }

    #[repr(C)]
    pub struct POINT {
        pub x: i32,
        pub y: i32,
    }

    #[repr(C)]
    pub struct MSG {
        pub hwnd: HWND,
        pub message: u32,
        pub wparam: WPARAM,
        pub lparam: LPARAM,
        pub time: u32,
        pub pt: POINT,
    }

    pub const CS_HREDRAW: u32 = 0x0001;
    pub const CS_VREDRAW: u32 = 0x0002;
    pub const WS_OVERLAPPEDWINDOW: u32 = 0x00CF_0000;
    pub const WS_VISIBLE: u32 = 0x1000_0000;
    pub const CW_USEDEFAULT: i32 = 0x8000_0000u32 as i32;
    pub const SW_SHOW: i32 = 5;
    pub const PM_REMOVE: u32 = 0x0001;

    pub type RegisterClassWFn = extern "system" fn(*const WNDCLASSW) -> u16;
    pub type CreateWindowExWFn = extern "system" fn(
        u32,
        *const u16,
        *const u16,
        u32,
        i32,
        i32,
        i32,
        i32,
        HWND,
        *mut c_void,
        HINSTANCE,
        *mut c_void,
    ) -> HWND;
    pub type DefWindowProcWn =
        extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT;
    pub type GetClientRectFn = extern "system" fn(HWND, *mut RECT) -> i32;
    pub type ShowWindowFn = extern "system" fn(HWND, i32) -> i32;
    pub type UpdateWindowFn = extern "system" fn(HWND) -> i32;
    pub type PeekMessageWFn =
        extern "system" fn(*mut MSG, HWND, u32, u32, u32) -> i32;
    pub type TranslateMessageFn = extern "system" fn(*const MSG) -> i32;
    pub type DispatchMessageWFn = extern "system" fn(*const MSG) -> LRESULT;

    pub struct User32 {
        pub register_class_w: RegisterClassWFn,
        pub create_window_ex_w: CreateWindowExWFn,
        pub get_client_rect: GetClientRectFn,
        pub show_window: ShowWindowFn,
        pub update_window: UpdateWindowFn,
        pub peek_message_w: PeekMessageWFn,
        pub translate_message: TranslateMessageFn,
        pub dispatch_message_w: DispatchMessageWFn,
    }

    static mut DEF_PROC: Option<DefWindowProcWn> = None;

    extern "system" fn wnd_proc(
        hwnd: HWND,
        msg: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        // SAFETY: DEF_PROC is resolved by `load()` before the window is created;
        // the window procedure runs on this single test thread, so there is no
        // concurrent mutation.
        let Some(def) = (unsafe { DEF_PROC }) else { return 0 };
        def(hwnd, msg, wparam, lparam)
    }

    pub fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain([0u16]).collect()
    }

    /// Loads user32.dll and resolves every entry this test needs.
    pub fn load() -> Result<User32, String> {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
        // SAFETY: a NUL-terminated literal.
        let module = unsafe { LoadLibraryA(b"user32.dll\0".as_ptr()) };
        if module.is_null() {
            return Err("LoadLibraryA(user32.dll)".into());
        }
        let resolve = |name: &[u8]| -> *const core::ffi::c_void {
            // SAFETY: module is loaded; name is NUL-terminated.
            unsafe { GetProcAddress(module as _, name.as_ptr()) }.map_or(
                core::ptr::null(),
                |f| f as *const core::ffi::c_void,
            )
        };
        // SAFETY: each pointer has the declared extern "system" signature.
        unsafe {
            let def: DefWindowProcWn = core::mem::transmute(resolve(b"DefWindowProcW\0"));
            DEF_PROC = Some(def);
            Ok(User32 {
                register_class_w: core::mem::transmute(resolve(b"RegisterClassW\0")),
                create_window_ex_w: core::mem::transmute(resolve(b"CreateWindowExW\0")),
                get_client_rect: core::mem::transmute(resolve(b"GetClientRect\0")),
                show_window: core::mem::transmute(resolve(b"ShowWindow\0")),
                update_window: core::mem::transmute(resolve(b"UpdateWindow\0")),
                peek_message_w: core::mem::transmute(resolve(b"PeekMessageW\0")),
                translate_message: core::mem::transmute(resolve(b"TranslateMessage\0")),
                dispatch_message_w: core::mem::transmute(resolve(b"DispatchMessageW\0")),
            })
        }
    }

    /// Registers the test class and creates the overlapped window.
    pub fn create_window(u32: &User32, class_name: &[u16], title: &[u16]) -> Result<HWND, String> {
        let class = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfn_wnd_proc: Some(wnd_proc),
            cb_cls_extra: 0,
            cb_wnd_extra: 0,
            h_instance: core::ptr::null_mut(),
            h_icon: core::ptr::null_mut(),
            h_cursor: core::ptr::null_mut(),
            hbr_background: core::ptr::null_mut(),
            lpsz_menu_name: core::ptr::null(),
            lpsz_class_name: class_name.as_ptr(),
        };
        // SAFETY: class is valid for the call.
        let atom = (u32.register_class_w)(&class);
        if atom == 0 {
            return Err("RegisterClassW failed".into());
        }
        // SAFETY: class/title are NUL-terminated; null parent/menu/param.
        let hwnd = (u32.create_window_ex_w)(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            640,
            480,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        if hwnd.is_null() {
            return Err("CreateWindowExW failed".into());
        }
        (u32.show_window)(hwnd, SW_SHOW);
        (u32.update_window)(hwnd);
        Ok(hwnd)
    }

    /// Drains pending messages without blocking.
    pub fn pump(u32: &User32, hwnd: HWND) {
        let mut msg = MSG {
            hwnd: core::ptr::null_mut(),
            message: 0,
            wparam: 0,
            lparam: 0,
            time: 0,
            pt: POINT { x: 0, y: 0 },
        };
        // SAFETY: msg is writable; PM_REMOVE drains the queue.
        while (u32.peek_message_w)(&mut msg, hwnd, 0, 0, PM_REMOVE) != 0 {
            (u32.translate_message)(&msg);
            (u32.dispatch_message_w)(&msg);
        }
    }
}

#[cfg(windows)]
fn pixel_format_name(fmt: victory_metal::sys::WMTPixelFormat) -> &'static str {
    use victory_metal::sys::WMTPixelFormat as F;
    match fmt {
        F::WMTPixelFormatBGRA8Unorm => "BGRA8Unorm",
        F::WMTPixelFormatBGRA8Unorm_sRGB => "BGRA8Unorm_sRGB",
        F::WMTPixelFormatRGBA8Unorm => "RGBA8Unorm",
        F::WMTPixelFormatRGBA8Unorm_sRGB => "RGBA8Unorm_sRGB",
        F::WMTPixelFormatRGB10A2Unorm => "RGB10A2Unorm",
        F::WMTPixelFormatRGBA16Float => "RGBA16Float",
        _ => "unknown",
    }
}

#[cfg(windows)]
fn run() -> Result<(), Box<dyn std::error::Error>> {
    use victory_metal::present::MetalLayer;
    use victory_metal::sys;
    use victory_metal::{Device, Pool};

    const FRAMES: usize = 120;

    let _pool = Pool::new();
    let device = Device::first()?;
    println!("device: {}", device.name());

    let u32 = win32::load()?;
    let class_name = win32::wide("F18PresentWindow");
    let title = win32::wide("F18 present test");
    let hwnd = win32::create_window(&u32, &class_name, &title)?;
    println!("hwnd: {hwnd:?}");
    win32::pump(&u32, hwnd);

    // Client area in window units; this becomes the layer drawable size.
    let mut rect = win32::RECT { left: 0, top: 0, right: 0, bottom: 0 };
    // SAFETY: rect is writable; hwnd is a live window.
    if (u32.get_client_rect)(hwnd, &mut rect) == 0 {
        return Err("GetClientRect failed".into());
    }
    let client_w = (rect.right - rect.left) as f64;
    let client_h = (rect.bottom - rect.top) as f64;
    println!("client rect: {client_w} x {client_h}");

    let layer = MetalLayer::for_window(&device, hwnd as isize)?;
    println!("CreateMetalViewFromHWND: OK (layer={:#x})", layer.handle());
    layer.set_drawable_size(client_w, client_h)?;
    layer.set_color_space(sys::WMTColorSpace::WMTColorSpaceSRGB)?;

    let (dw, dh) = layer.drawable_size()?;
    println!("layer drawable size (getProps): {dw} x {dh}");

    let queue = device.new_queue(4)?;

    let mut frame_times: Vec<std::time::Duration> = Vec::with_capacity(FRAMES);
    let mut tex_w = 0u64;
    let mut tex_h = 0u64;
    let mut tex_fmt = sys::WMTPixelFormat::WMTPixelFormatInvalid;

    for i in 0..FRAMES {
        let t0 = std::time::Instant::now();

        let cb = queue.command_buffer()?;
        let drawable = layer.next_drawable()?;
        let texture = drawable.texture()?;

        if i == 0 {
            // SAFETY: a live texture from the drawable.
            tex_w = unsafe { sys::MTLTexture_width(texture.handle()) };
            tex_h = unsafe { sys::MTLTexture_height(texture.handle()) };
            tex_fmt = unsafe { sys::MTLTexture_pixelFormat(texture.handle()) };
            println!(
                "drawable texture: {tex_w} x {tex_h}, pixel format {} ({})",
                tex_fmt.0,
                pixel_format_name(tex_fmt)
            );
        }

        let t = i as f64 / (FRAMES - 1) as f64;
        let clear = sys::WMTClearColor {
            r: t,
            g: 1.0 - t,
            b: 0.3,
            a: 1.0,
        };

        let mut info: sys::WMTRenderPassInfo = unsafe { std::mem::zeroed() };
        info.colors[0].texture = texture.handle();
        info.colors[0].load_action = sys::WMTLoadAction::WMTLoadActionClear;
        info.colors[0].store_action = sys::WMTStoreAction::WMTStoreActionStore;
        info.colors[0].clear_color = clear;
        info.render_target_width = tex_w as u32;
        info.render_target_height = tex_h as u32;

        // SAFETY: the command buffer and pass info are live for the call.
        let encoder = unsafe { sys::MTLCommandBuffer_renderCommandEncoder(cb.handle(), &mut info) };
        if encoder == 0 {
            return Err("renderCommandEncoder returned null".into());
        }
        // SAFETY: a live encoder; clear-only pass needs no command chain.
        unsafe { sys::MTLCommandEncoder_endEncoding(encoder) };

        cb.present(&drawable);
        cb.commit();
        cb.wait()?;

        frame_times.push(t0.elapsed());
        win32::pump(&u32, hwnd);
    }

    let mut sorted = frame_times.clone();
    sorted.sort();
    let min = sorted.first().copied().unwrap_or_default();
    let max = sorted.last().copied().unwrap_or_default();
    let median = sorted[sorted.len() / 2];
    let total: std::time::Duration = frame_times.iter().sum();
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;

    println!(
        "frames: {} total: {:.3} ms  min: {:.3} ms  median: {:.3} ms  max: {:.3} ms",
        FRAMES,
        ms(total),
        ms(min),
        ms(median),
        ms(max)
    );
    println!(
        "final drawable texture: {tex_w} x {tex_h}, pixel format {} ({})",
        tex_fmt.0,
        pixel_format_name(tex_fmt)
    );

    // Leave the window on screen for a while so it can be captured.
    std::thread::sleep(std::time::Duration::from_secs(12));
    win32::pump(&u32, hwnd);

    // Explicitly drop the layer (releases the MetalView) before exiting.
    drop(layer);
    Ok(())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("victory-metal-present runs on Windows (Wine) only");
}
