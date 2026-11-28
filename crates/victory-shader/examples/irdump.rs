//! A container's program as the translator sees it: blocks (offset, predecessors, exit as the
//! lanes take it) and their instructions in SP3 syntax.
//!
//! `cargo run -p victory-shader --example irdump -- <file.cdxil>`
use victory_shader::structure::LaneExit;
use victory_shader::{Container, FourCC};

fn main() {
    let path = std::env::args().nth(1).expect("container");
    let bytes = std::fs::read(&path).expect("read");
    let c = Container::parse(&bytes).expect("container");
    let dxil = c.part(FourCC::DXIL).expect("DXIL part");
    let native = victory_shader::native_programs(dxil.data)
        .expect("native programs")
        .into_iter()
        .find(|p| !p.code.is_empty())
        .expect("a program");
    println!("used={:#?}", native.used);
    println!("layout={:?}", native.layout);
    let p = victory_shader::spill::build(&native).expect("ir");
    let st = victory_shader::structure::structure(&p);
    if let Err(e) = &st {
        println!("structure: {e}");
    }
    for (b, blk) in p.blocks.iter().enumerate() {
        let exit = match st.as_ref().ok().map(|s| &s.exits[b]) {
            Some(LaneExit::Branch {
                when,
                taken,
                not_taken,
            }) => {
                let wide = st.as_ref().unwrap().wide.contains(&b);
                format!(
                    "branch{} taken={taken} not={not_taken} when={when:?}",
                    if wide { "(wide)" } else { "" }
                )
            }
            Some(e) => format!("{e:?}"),
            None => format!("{:?}", blk.exit),
        };
        println!(
            "== block {b} @{:#x} preds={:?} -> {exit}",
            blk.offset, blk.preds
        );
        for i in &blk.insts {
            println!("  {:#06x} {}", i.machine.offset, i.machine.text(p.wave64));
        }
    }
}
