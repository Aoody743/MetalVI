use super::ListExpansion;
use crate::host::{HostList, host, transition};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

const SIDE: u32 = 16;
const PITCH: u32 = 256;
const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

fn shader(name: &str) -> Vec<u8> {
    let path = std::env::var_os(name).unwrap_or_else(|| panic!("set {name} to the compiled DXIL"));
    std::fs::read(path).unwrap()
}

fn code(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    }
}

fn vertex_view<T: Copy>(list: &HostList, values: &[T]) -> D3D12_VERTEX_BUFFER_VIEW {
    let bytes = size_of_val(values);
    let (gpu, cpu) = list.scratch(bytes as u64).unwrap();
    // SAFETY: list scratch is writable for bytes and aligned for our data.
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), cpu.cast(), values.len()) };
    D3D12_VERTEX_BUFFER_VIEW {
        BufferLocation: gpu,
        SizeInBytes: bytes as u32,
        StrideInBytes: size_of::<T>() as u32,
    }
}

fn resource(
    device: &ID3D12Device,
    kind: D3D12_HEAP_TYPE,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: kind,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut out = None;
    // SAFETY: descriptions are complete and out is writable.
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, desc, state, None, &mut out)
            .unwrap()
    };
    out.unwrap()
}

#[test]
#[ignore = "requires D3D12 and F57_VS/F57_PS compiled shader files"]
fn console_lists_render_to_a_d3d12_target() {
    let host = host().expect("D3D12 device unavailable");
    let device = &host.device;
    let (vs, ps) = (shader("F57_VS"), shader("F57_PS"));
    let signature = D3D12_ROOT_SIGNATURE_DESC {
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    let sample = DXGI_SAMPLE_DESC {
        Count: 1,
        Quality: 0,
    };
    // SAFETY: the descriptors, bytecodes, resources and command lists all
    // outlive their API calls; host.execute waits before CPU readback.
    unsafe {
        let mut root_blob: Option<ID3DBlob> = None;
        D3D12SerializeRootSignature(
            &signature,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut root_blob,
            None,
        )
        .unwrap();
        let root_blob = root_blob.unwrap();
        let root: ID3D12RootSignature = device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(
                    root_blob.GetBufferPointer().cast(),
                    root_blob.GetBufferSize(),
                ),
            )
            .unwrap();
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: 15,
            ..Default::default()
        };
        let face = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        let inputs = [
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("POSITION"),
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                ..Default::default()
            },
            D3D12_INPUT_ELEMENT_DESC {
                SemanticName: s!("INSTANCE"),
                Format: DXGI_FORMAT_R32_UINT,
                InputSlot: 1,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                InstanceDataStepRate: 1,
                ..Default::default()
            },
        ];
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: code(&vs),
            PS: code(&ps),
            BlendState: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                FrontCounterClockwise: true.into(),
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                FrontFace: face,
                BackFace: face,
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: inputs.as_ptr(),
                NumElements: inputs.len() as u32,
            },
            NumRenderTargets: 1,
            RTVFormats: [
                FORMAT,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
            ],
            SampleDesc: sample,
            ..Default::default()
        };
        let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
        let readback = resource(
            device,
            D3D12_HEAP_TYPE_READBACK,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(PITCH * SIDE),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: sample,
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        for topology in [
            ListExpansion::Quads,
            ListExpansion::Fan,
            ListExpansion::QuadStrip,
        ] {
            let color = resource(
                device,
                D3D12_HEAP_TYPE_DEFAULT,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(SIDE),
                    Height: SIDE,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: sample,
                    Format: FORMAT,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            let list = host.new_list().unwrap();
            let rtv = host.rtv(Some(&color), None);
            let l = &list.list;
            let mut source = [0u16; 5];
            let (view, base) = if topology == ListExpansion::Quads {
                (topology.generated_view(&list, 4).unwrap(), 2)
            } else {
                (
                    topology
                        .title_view(&list, source.as_ptr() as u64, 10, 57, 1, 4)
                        .unwrap(),
                    -3,
                )
            };
            let tl = [-0.5f32, 0.5];
            let bl = [-0.5, -0.5];
            let br = [0.5, -0.5];
            let tr = [0.5, 0.5];
            let vertices = [[2.0; 2], [2.0; 2], tl, bl, br, tr, tl, bl, tr, br];
            l.IASetVertexBuffers(
                0,
                Some(&[
                    vertex_view(&list, &vertices),
                    vertex_view(&list, &[0u32, 1, 2, 3, 4]),
                ]),
            );
            l.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, None);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIDE as f32,
                Height: SIDE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                right: SIDE as i32,
                bottom: SIDE as i32,
                ..Default::default()
            }]);
            l.SetGraphicsRootSignature(&root);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(topology.host_topology() as i32));
            l.IASetIndexBuffer(Some(&view));
            l.DrawIndexedInstanced(topology.index_count(4).unwrap(), 1, 0, base, 4);
            l.ResourceBarrier(&[transition(
                &color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&color),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: FORMAT,
                            Width: SIDE,
                            Height: SIDE,
                            Depth: 1,
                            RowPitch: PITCH,
                        },
                    },
                },
            };
            l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.close();
            // These indices do not exist when title_view records its upload.
            source.copy_from_slice(if topology == ListExpansion::Fan {
                &[0, 5, 6, 7, 8]
            } else {
                &[0, 9, 10, 11, 12]
            });
            host.execute(&[&list]);
            std::hint::black_box(&source); // title_view reads at submission, not recording.
            let mut mapped = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), (SIDE * PITCH) as usize);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let i = (y * PITCH + x * 4) as usize;
                    let inside = (4..12).contains(&x) && (4..12).contains(&y);
                    assert_eq!(
                        &bytes[i..i + 4],
                        if inside {
                            &[0, 255, 0, 255]
                        } else {
                            &[0, 0, 0, 255]
                        },
                        "{topology:?} at {x},{y}"
                    );
                }
            }
            readback.Unmap(0, None);
            println!("{topology:?}: {} pixels matched", SIDE * SIDE);
        }
    }
}
