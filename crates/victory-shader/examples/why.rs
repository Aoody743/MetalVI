//! One container's translation with the wave reasons (`VICTORY_WAVE_WHY`) on a 32-lane host.
fn main() {
    let wave: u32 = std::env::var("SURVEY_HOST_WAVE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    victory_shader::emit::set_host_max_wave(wave);
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).expect("read");
        eprintln!("=== {path}");
        match victory_shader::emit::translate(&bytes) {
            Ok(h) => eprintln!(
                "RESULT ok dxil_bytes={} wave={:?}",
                h.dxil.len(),
                victory_shader::emit::last_wave()
            ),
            Err(e) => eprintln!("RESULT err {e}"),
        }
    }
}
