//! GPU renderer (wgpu): buffers and a line stage on top of them
//! (05-architecture, §7).
//!
//! 1. Geometry: every part as a 3D body → part ID and depth per pixel
//!    (supersampled for anti-aliasing).
//! 2. Edges: a line seed where the part changes, except a seam between parts
//!    of one skin with continuous depth. The nearer part owns the line.
//! 3. Jump flood: every pixel learns its nearest seed.
//! 4. Compose: fills, ground with grid and shadows, lines with width by depth.
//!
//! Layers can be switched off: paint (fills) and lines.

pub use mavian_frame::{Layers, Renderer, SS};

use mavian_frame::{CamU, FillVertex, STEPS, SceneU, Segment, Vertex, build, faces, strokes, uniforms};
use mavian_scene::{FrameState, Shot};
use wgpu::util::DeviceExt;





#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CurvePoint {
    pos: [f32; 4],
    id: u32,
    _pad: [u32; 3],
}


struct Targets {
    size: (u32, u32),
    depth: wgpu::TextureView,
    gid: wgpu::TextureView,
    gdepth: wgpu::TextureView,
    gid_top: wgpu::TextureView,
    gdepth_top: wgpu::TextureView,
    gid_contacts: wgpu::TextureView,
    gdepth_contacts: wgpu::TextureView,
    strength: wgpu::TextureView,
    strokes: wgpu::TextureView,
    seeds: [wgpu::TextureView; 2],
    out: wgpu::Texture,
    readback: wgpu::Buffer,
    row: u32,
}

/// Version of the GPU drawing: bump on any edit that changes the picture.
pub const VERSION: &str = "gpu-43";

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// adapter name: different GPUs may differ in the last bits of a pixel
    pub adapter: String,
    geo: wgpu::RenderPipeline,
    geo_top: wgpu::RenderPipeline,
    geo_contacts: wgpu::RenderPipeline,
    edge: wgpu::RenderPipeline,
    jfa: wgpu::RenderPipeline,
    compose: wgpu::RenderPipeline,
    visible: wgpu::ComputePipeline,
    effect_depth: wgpu::ComputePipeline,
    stroke: wgpu::RenderPipeline,
    fill: wgpu::RenderPipeline,
    cam: wgpu::Buffer,
    scene: wgpu::Buffer,
    steps: Vec<wgpu::Buffer>,
    targets: Option<Targets>,
}

fn attach(view: &wgpu::TextureView, clear: f64) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color { r: clear, g: 0.0, b: 0.0, a: 0.0 }),
            store: wgpu::StoreOp::Store,
        },
    })
}

fn group_compute(device: &wgpu::Device, p: &wgpu::ComputePipeline, entries: &[(u32, wgpu::BindingResource)]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = entries.iter().map(|(b, r)| wgpu::BindGroupEntry { binding: *b, resource: r.clone() }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries })
}



impl Gpu {
    /// Open the GPU. `low_power` — prefer the integrated one (laptops).
    pub fn new(low_power: bool) -> Result<Gpu, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: if low_power { wgpu::PowerPreference::LowPower } else { wgpu::PowerPreference::HighPerformance },
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mavian"),
            ..Default::default()
        }))
        .map_err(|e| format!("cannot open the GPU: {e}"))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mavian"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let target = |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let pipeline = |label: &str, vs: &str, fs: &str, buffers: &[Option<wgpu::VertexBufferLayout>], targets: &[Option<wgpu::ColorTargetState>], depth: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth.then(|| wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let vertex = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32],
        };
        use wgpu::TextureFormat as F;
        let geo = pipeline("geometry", "vs_geo", "fs_geo", &[Some(vertex.clone())], &[target(F::R32Uint), target(F::R32Float)], true);
        let geo_top = pipeline("geometry on top", "vs_geo", "fs_geo_top", &[Some(vertex.clone())], &[target(F::R32Uint), target(F::R32Float)], true);
        let geo_contacts = pipeline("geometry on top", "vs_geo", "fs_geo_contacts", &[Some(vertex)], &[target(F::R32Uint), target(F::R32Float)], true);
        let edge = pipeline("edges", "vs_full", "fs_edge", &[], &[target(F::R32Uint), target(F::R32Float)], false);
        let jfa = pipeline("jump flood", "vs_full", "fs_jfa", &[], &[target(F::R32Uint)], false);
        let compose = pipeline("compose", "vs_full", "fs_compose", &[], &[target(F::Rgba8Unorm)], false);
        let visible = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("curve visibility"),
            layout: None,
            module: &module,
            entry_point: Some("cs_visible"),
            compilation_options: Default::default(),
            cache: None,
        });
        // strokes of one colour overlap without darkening: keep the strongest
        let max = wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Max };
        let segment = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Segment>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x4, 4 => Float32x2, 5 => Uint32, 6 => Uint32],
        };
        let stroke = pipeline(
            "strokes",
            "vs_stroke",
            "fs_stroke",
            &[Some(segment)],
            &[Some(wgpu::ColorTargetState {
                format: F::Rgba16Float,
                blend: Some(wgpu::BlendState { color: max, alpha: max }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            false,
        );
        let fill_vertex = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FillVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4, 2 => Uint32],
        };
        let fill = pipeline(
            "fills",
            "vs_fill",
            "fs_fill",
            &[Some(fill_vertex)],
            &[Some(wgpu::ColorTargetState {
                format: F::Rgba16Float,
                blend: Some(wgpu::BlendState { color: max, alpha: max }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            false,
        );
        let uniform = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cam = uniform("camera", std::mem::size_of::<CamU>() as u64);
        let scene = uniform("scene", std::mem::size_of::<SceneU>() as u64);
        let steps = STEPS
            .iter()
            .map(|&s| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("jump flood step"),
                    contents: bytemuck::cast_slice(&[s, 0, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();
        let effect_depth = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("effect depth"), layout: None, module: &module,
            entry_point: Some("cs_effect_depth"), compilation_options: Default::default(), cache: None,
        });
        Ok(Gpu {
            device,
            queue,
            adapter: format!("{} ({})", info.name, info.driver_info),
            geo,
            geo_top,
            geo_contacts,
            edge,
            jfa,
            compose,
            visible,
            effect_depth,
            stroke,
            fill,
            cam,
            scene,
            steps,
            targets: None,
        })
    }

    fn targets(&mut self, size: (u32, u32)) -> &Targets {
        if self.targets.as_ref().is_none_or(|t| t.size != size) {
            let (w, h) = (size.0 * SS, size.1 * SS);
            let tex = |label: &str, format, w: u32, h: u32, extra: wgpu::TextureUsages| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | extra,
                    view_formats: &[],
                })
            };
            let view = |t: wgpu::Texture| t.create_view(&Default::default());
            use wgpu::{TextureFormat as F, TextureUsages as U};
            let row = (size.0 * 4).div_ceil(256) * 256;
            self.targets = Some(Targets {
                size,
                depth: view(tex("depth", F::Depth32Float, w, h, U::empty())),
                gid: view(tex("part id", F::R32Uint, w, h, U::TEXTURE_BINDING)),
                gdepth: view(tex("depth value", F::R32Float, w, h, U::TEXTURE_BINDING)),
                gid_top: view(tex("part id on top", F::R32Uint, w, h, U::TEXTURE_BINDING)),
                gdepth_top: view(tex("depth value on top", F::R32Float, w, h, U::TEXTURE_BINDING)),
                gid_contacts: view(tex("contact id", F::R32Uint,w,h,U::TEXTURE_BINDING)),
                gdepth_contacts: view(tex("contact depth", F::R32Float,w,h,U::TEXTURE_BINDING)),
                strength: view(tex("line strength", F::R32Float, w, h, U::TEXTURE_BINDING)),
                strokes: view(tex("strokes", F::Rgba16Float, w, h, U::TEXTURE_BINDING)),
                seeds: [
                    view(tex("seeds a", F::R32Uint, w, h, U::TEXTURE_BINDING)),
                    view(tex("seeds b", F::R32Uint, w, h, U::TEXTURE_BINDING)),
                ],
                out: tex("frame", F::Rgba8Unorm, size.0, size.1, U::COPY_SRC),
                readback: self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("readback"),
                    size: (row * size.1) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                row,
            });
        }
        self.targets.as_ref().expect("made above")
    }

    /// Draw frame `f` of `shot` → RGBA pixels, `size.0 × size.1 × 4` bytes.
    pub fn render(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> Vec<u8> {
        self.render_frame(shot,f,layers,false).rgba
    }

    pub fn render_with_effects(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> (Vec<u8>, String) {
        let frame=self.render_with_depth(shot,f,layers);
        let svg=frame.fires_svg(shot,f);
        (frame.rgba,svg)
    }

    pub fn render_with_depth(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> mavian_frame::RenderedFrame {
        self.render_frame(shot,f,layers,true)
    }

    fn render_frame(&mut self, shot: &Shot, f: &FrameState, layers: Layers, effects: bool) -> mavian_frame::RenderedFrame {
        let size = shot.size;
        let mesh = build(shot, f);
        let c = &f.cam;
        let (cam, scene) = uniforms(shot, f, &mesh, layers);
        self.queue.write_buffer(&self.cam, 0, bytemuck::bytes_of(&cam));
        self.queue.write_buffer(&self.scene, 0, bytemuck::bytes_of(&scene));
        let parts = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("parts"),
            contents: bytemuck::cast_slice(&mesh.parts),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let upload = |v: &[Vertex]| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertices"),
                contents: bytemuck::cast_slice(if v.is_empty() { &[Vertex { pos: [0.0; 3], id: 0 }][..] } else { v }),
                usage: wgpu::BufferUsages::VERTEX,
            })
        };
        let (vertices, top, contacts) = (upload(&mesh.vertices), upload(&mesh.top), upload(&mesh.contacts));
        self.targets(size);
        let t = self.targets.as_ref().expect("made above");
        let device = &self.device;
        let group = |p: &wgpu::RenderPipeline, entries: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<wgpu::BindGroupEntry> =
                entries.iter().map(|(b, r)| wgpu::BindGroupEntry { binding: *b, resource: r.clone() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries })
        };
        use wgpu::BindingResource as R;
        let cam_r = || self.cam.as_entire_binding();
        let geo_bg = group(&self.geo, &[(0, cam_r())]);
        let first = || {
            vec![(0, cam_r()), (1, parts.as_entire_binding()), (2, R::TextureView(&t.gid)), (3, R::TextureView(&t.gdepth))]
        };
        let actor_buffers = || {
            let mut v = first();
            v.extend([(9, R::TextureView(&t.gid_top)), (10, R::TextureView(&t.gdepth_top))]);
            v
        };
        let both = || {
            let mut v=actor_buffers();v.extend([(16,R::TextureView(&t.gid_contacts)),(17,R::TextureView(&t.gdepth_contacts))]);v
        };
        let contacts_bg=group(&self.geo_contacts,&actor_buffers());
        let top_bg = group(&self.geo_top, &first());
        let edge_bg = group(&self.edge, &both());
        let jfa_bgs: Vec<_> = (0..STEPS.len())
            .map(|k| group(&self.jfa, &[(0, cam_r()), (4, R::TextureView(&t.seeds[k % 2])), (5, self.steps[k].as_entire_binding())]))
            .collect();
        let last = STEPS.len() % 2;
        let mut compose_entries = both();
        compose_entries.extend([
            (4, R::TextureView(&t.seeds[last])),
            (6, self.scene.as_entire_binding()),
            (11, R::TextureView(&t.strength)),
            (14, R::TextureView(&t.strokes)),
        ]);
        // curve points for the visibility test
        let points: Vec<CurvePoint> = mesh
            .curves
            .iter()
            .flat_map(|c| c.points.iter().map(move |p| CurvePoint { pos: [p.x as f32, p.y as f32, p.z as f32, 1.0], id: c.id, _pad: [0; 3] }))
            .collect();
        let n_points = points.len();
        let points_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("curve points"),
            contents: bytemuck::cast_slice(if points.is_empty() { &[CurvePoint { pos: [0.0; 4], id: 0, _pad: [0; 3] }][..] } else { &points }),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let vis_size = (n_points.max(1) * 4) as u64;
        let vis_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("curve visibility"),
            size: vis_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let vis_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("curve visibility readback"),
            size: vis_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut vis_entries = both();
        vis_entries.extend([(12, points_buf.as_entire_binding()), (13, vis_buf.as_entire_binding())]);
        let vis_bg = group_compute(device, &self.visible, &vis_entries);
        let depth_export = (effects && !shot.fires.is_empty()).then(|| {
            let bytes = (size.0 as u64 * size.1 as u64 * SS as u64 * SS as u64)*4;
            let output = device.create_buffer(&wgpu::BufferDescriptor { label:Some("effect depth"),size:bytes,usage:wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false });
            let read = device.create_buffer(&wgpu::BufferDescriptor { label:Some("effect depth readback"),size:bytes,usage:wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,mapped_at_creation:false });
            let bg = group_compute(device,&self.effect_depth,&[(0,cam_r()),(3,R::TextureView(&t.gdepth)),(9,R::TextureView(&t.gid_top)),(10,R::TextureView(&t.gdepth_top)),(15,output.as_entire_binding()),(16,R::TextureView(&t.gid_contacts)),(17,R::TextureView(&t.gdepth_contacts))]);
            (output,read,bg,bytes)
        });
        let compose_bg = group(&self.compose, &compose_entries);
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("geometry"),
                color_attachments: &[attach(&t.gid, 0.0), attach(&t.gdepth, 1e9)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.geo);
            pass.set_bind_group(0, &geo_bg, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.draw(0..mesh.vertices.len() as u32, 0..1);
        }
        {
            // parts on top by a rule: their own depth test among themselves
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("geometry on top"),
                color_attachments: &[attach(&t.gid_top, 0.0), attach(&t.gdepth_top, 1e9)],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.geo_top);
            pass.set_bind_group(0, &top_bg, &[]);
            pass.set_vertex_buffer(0, top.slice(..));
            pass.draw(0..mesh.top.len() as u32, 0..1);
        }
        {
            let mut pass=enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label:Some("contact props"),color_attachments:&[attach(&t.gid_contacts,0.0),attach(&t.gdepth_contacts,1e9)],
                depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment { view:&t.depth,depth_ops:Some(wgpu::Operations { load:wgpu::LoadOp::Clear(1.0),store:wgpu::StoreOp::Store }),stencil_ops:None }),
                ..Default::default()
            });
            pass.set_pipeline(&self.geo_contacts);pass.set_bind_group(0,&contacts_bg,&[]);
            pass.set_vertex_buffer(0,contacts.slice(..));pass.draw(0..mesh.contacts.len() as u32,0..1);
        }
        let mut full = |pipe: &wgpu::RenderPipeline, bg: &wgpu::BindGroup, views: &[Option<wgpu::RenderPassColorAttachment>]| {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: views,
                ..Default::default()
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        };
        let none = u32::MAX as f64;
        full(&self.edge, &edge_bg, &[attach(&t.seeds[0], none), attach(&t.strength, 0.0)]);
        for (k, bg) in jfa_bgs.iter().enumerate() {
            full(&self.jfa, bg, &[attach(&t.seeds[(k + 1) % 2], none)]);
        }
        drop(full);
        if n_points > 0 && layers.lines {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.visible);
            pass.set_bind_group(0, &vis_bg, &[]);
            pass.dispatch_workgroups((n_points as u32).div_ceil(64), 1, 1);
        }
        if let Some((output,read,bg,bytes)) = &depth_export {
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.effect_depth);pass.set_bind_group(0,bg,&[]);
                pass.dispatch_workgroups((size.0*SS).div_ceil(8),(size.1*SS).div_ceil(8),1);
            }
            enc.copy_buffer_to_buffer(output,0,read,0,*bytes);
        }
        enc.copy_buffer_to_buffer(&vis_buf, 0, &vis_read, 0, vis_size);
        self.queue.submit([enc.finish()]);
        let vslice = vis_read.slice(..);
        vslice.map_async(wgpu::MapMode::Read, |r| r.expect("map visibility"));
        if let Some((_,read,_,_)) = &depth_export { read.slice(..).map_async(wgpu::MapMode::Read,|r|r.expect("map effect depth")); }
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("GPU wait");
        let vis: Vec<f32> = bytemuck::cast_slice(&vslice.get_mapped_range().expect("read visibility")).to_vec();
        vis_read.unmap();
        let fire_depth=depth_export.as_ref().map(|(_,read,_,_)| {
            let values={
                let mapped=read.slice(..).get_mapped_range().expect("read effect depth");
                bytemuck::cast_slice::<u8,f32>(&mapped).to_vec()
            };
            read.unmap();values
        });
        let (mut segments, fills) = if layers.lines { faces(&mesh.faces) } else { (Vec::new(), Vec::new()) };
        if layers.lines {
            segments.extend(strokes(&mesh.curves, &vis, c));
        }
        let fill_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fills"),
            contents: bytemuck::cast_slice(if fills.is_empty() { &[FillVertex { pos: [0.0; 2], color: [0.0; 4], id: 0 }][..] } else { &fills }),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let fill_bg = group(&self.fill, &[(0, cam_r()), (2, R::TextureView(&t.gid)), (9, R::TextureView(&t.gid_top)), (16,R::TextureView(&t.gid_contacts))]);
        let seg_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stroke segments"),
            contents: bytemuck::cast_slice(if segments.is_empty() { &[Segment { a: [0.0; 2], b: [0.0; 2], r: [0.0; 2], color: [0.0; 4], depth: [0.0; 2], id: 0, hidden: 0 }][..] } else { &segments }),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let stroke_bg = group(&self.stroke, &both());
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("strokes"),
                color_attachments: &[attach(&t.strokes, 0.0)],
                ..Default::default()
            });
            pass.set_pipeline(&self.stroke);
            pass.set_bind_group(0, &stroke_bg, &[]);
            pass.set_vertex_buffer(0, seg_buf.slice(..));
            pass.draw(0..6, 0..segments.len() as u32);
            pass.set_pipeline(&self.fill);
            pass.set_bind_group(0, &fill_bg, &[]);
            pass.set_vertex_buffer(0, fill_buf.slice(..));
            pass.draw(0..fills.len() as u32, 0..1);
        }
        let out_view = t.out.create_view(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("compose"),
                color_attachments: &[attach(&out_view, 0.0)],
                ..Default::default()
            });
            pass.set_pipeline(&self.compose);
            pass.set_bind_group(0, &compose_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &t.out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &t.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(t.row), rows_per_image: Some(size.1) },
            },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = t.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map the frame"));
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).expect("GPU wait");
        let mut px = Vec::with_capacity((size.0 * size.1 * 4) as usize);
        {
            let data = slice.get_mapped_range().expect("read the frame");
            for y in 0..size.1 as usize {
                let start = y * t.row as usize;
                px.extend_from_slice(&data[start..start + size.0 as usize * 4]);
            }
        }
        t.readback.unmap();
        mavian_frame::RenderedFrame { rgba:px,fire_depth }
    }
}

impl Renderer for Gpu {
    fn render(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> Vec<u8> {
        Gpu::render(self, shot, f, layers)
    }

    fn render_with_depth(&mut self, shot: &Shot, f: &FrameState, layers: Layers) -> mavian_frame::RenderedFrame {
        Gpu::render_with_depth(self,shot,f,layers)
    }
    fn device(&self) -> String {
        format!("{} {}", VERSION, self.adapter)
    }
}
