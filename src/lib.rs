use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, WindowEvent}, 
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window},
};

#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowExtWebSys;

use cgmath::{perspective, Deg, InnerSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

const COLOR_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const COLOR_YELLOW: [f32; 3] = [1.0, 1.0, 0.0];
const COLOR_BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const COLOR_GREEN: [f32; 3] = [0.0, 1.0, 0.0];
const COLOR_RED: [f32; 3] = [1.0, 0.0, 0.0];
const COLOR_ORANGE: [f32; 3] = [1.0, 0.647, 0.0];

const CUBE_POSITIONS: [[f32; 3]; 8] = [
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
];

const VERTICES: &[Vertex] = &[
    Vertex {
        position: CUBE_POSITIONS[0],
        color: COLOR_GREEN,
    },
    Vertex {
        position: CUBE_POSITIONS[1],
        color: COLOR_GREEN,
    },
    Vertex {
        position: CUBE_POSITIONS[2],
        color: COLOR_GREEN,
    },
    Vertex {
        position: CUBE_POSITIONS[3],
        color: COLOR_GREEN,
    },
    Vertex {
        position: CUBE_POSITIONS[5],
        color: COLOR_BLUE,
    },
    Vertex {
        position: CUBE_POSITIONS[4],
        color: COLOR_BLUE,
    },
    Vertex {
        position: CUBE_POSITIONS[7],
        color: COLOR_BLUE,
    },
    Vertex {
        position: CUBE_POSITIONS[6],
        color: COLOR_BLUE,
    },
    Vertex {
        position: CUBE_POSITIONS[3],
        color: COLOR_WHITE,
    },
    Vertex {
        position: CUBE_POSITIONS[2],
        color: COLOR_WHITE,
    },
    Vertex {
        position: CUBE_POSITIONS[6],
        color: COLOR_WHITE,
    },
    Vertex {
        position: CUBE_POSITIONS[7],
        color: COLOR_WHITE,
    },
    Vertex {
        position: CUBE_POSITIONS[4],
        color: COLOR_YELLOW,
    },
    Vertex {
        position: CUBE_POSITIONS[5],
        color: COLOR_YELLOW,
    },
    Vertex {
        position: CUBE_POSITIONS[1],
        color: COLOR_YELLOW,
    },
    Vertex {
        position: CUBE_POSITIONS[0],
        color: COLOR_YELLOW,
    },
    Vertex {
        position: CUBE_POSITIONS[1],
        color: COLOR_RED,
    },
    Vertex {
        position: CUBE_POSITIONS[5],
        color: COLOR_RED,
    },
    Vertex {
        position: CUBE_POSITIONS[6],
        color: COLOR_RED,
    },
    Vertex {
        position: CUBE_POSITIONS[2],
        color: COLOR_RED,
    },
    Vertex {
        position: CUBE_POSITIONS[4],
        color: COLOR_ORANGE,
    },
    Vertex {
        position: CUBE_POSITIONS[0],
        color: COLOR_ORANGE,
    },
    Vertex {
        position: CUBE_POSITIONS[3],
        color: COLOR_ORANGE,
    },
    Vertex {
        position: CUBE_POSITIONS[7],
        color: COLOR_ORANGE,
    },
];

const INDICES: &[u16] = &[
    0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7, 8, 9, 10, 8, 10, 11, 12, 13, 14, 12, 14, 15, 16, 17, 18,
    16, 18, 19, 20, 21, 22, 20, 22, 23,
];

#[derive(Debug)]
struct Camera {
    position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Camera {
    fn new<P: Into<Point3<f32>>, Y: Into<Rad<f32>>, PT: Into<Rad<f32>>>(
        position: P,
        yaw: Y,
        pitch: PT,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    fn calc_view_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();
        let target = self.position
            + Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize();
        Matrix4::look_at_rh(self.position, target, Vector3::unit_y())
    }
}

#[derive(Debug)]
struct CameraController {
    move_speed: f32,
    mouse_sensitivity: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
}

impl CameraController {
    fn new(move_speed: f32, mouse_sensitivity: f32) -> Self {
        Self {
            move_speed,
            mouse_sensitivity,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
        }
    }

    fn process_keyboard(&mut self, key_code: PhysicalKey, state: ElementState) -> bool {
        let is_pressed = state == ElementState::Pressed;
        match key_code {
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.is_forward_pressed = is_pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.is_backward_pressed = is_pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.is_left_pressed = is_pressed;
                true
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.is_right_pressed = is_pressed;
                true
            }
            PhysicalKey::Code(KeyCode::Space) => {
                self.is_up_pressed = is_pressed;
                true
            }
            PhysicalKey::Code(KeyCode::ShiftLeft) => {
                self.is_down_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    fn process_mouse_motion(&self, mouse_dx: f64, mouse_dy: f64, camera: &mut Camera) {
        camera.yaw += Rad(mouse_dx as f32 * self.mouse_sensitivity);
        camera.pitch -= Rad(mouse_dy as f32 * self.mouse_sensitivity);
        let eighty_nine_degrees: Rad<f32> = Deg(89.0).into();
        camera.pitch.0 = camera
            .pitch
            .0
            .clamp(-eighty_nine_degrees.0, eighty_nine_degrees.0);
    }

    fn update_camera_position(&self, camera: &mut Camera, dt: f32) {
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();

        if self.is_forward_pressed {
            camera.position += forward * self.move_speed * dt;
        }
        if self.is_backward_pressed {
            camera.position -= forward * self.move_speed * dt;
        }
        if self.is_right_pressed {
            camera.position += right * self.move_speed * dt;
        }
        if self.is_left_pressed {
            camera.position -= right * self.move_speed * dt;
        }
        if self.is_up_pressed {
            camera.position.y += self.move_speed * dt;
        }
        if self.is_down_pressed {
            camera.position.y -= self.move_speed * dt;
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            mvp: Matrix4::identity().into(),
        }
    }
    fn update_mvp(&mut self, camera: &Camera, aspect_ratio: f32, model_angle_rad: f32) {
        let projection = perspective(Deg(45.0), aspect_ratio, 0.1, 100.0);
        let view = camera.calc_view_matrix();
        let model = Matrix4::from_angle_y(Rad(model_angle_rad));
        self.mvp = (projection * view * model).into();
    }
}

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let desc = wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let texture = device.create_texture(&desc);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        compare: Some(wgpu::CompareFunction::LessEqual),
        lod_min_clamp: 0.0,
        lod_max_clamp: 100.0,
        ..Default::default()
    });
    (texture, view, sampler)
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    window: Arc<Window>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    owned_depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    _owned_depth_sampler: wgpu::Sampler,
    camera: Camera,
    camera_controller: CameraController,
    model_angle: f32,
    model_rotation_speed: f32,
    window_focused: bool,
    cursor_grab_toggled_on: bool,
    last_update: Instant,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance_descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
        };
        let instance = wgpu::Instance::new(&instance_descriptor);

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };
        let adapter = instance
            .request_adapter(&adapter_options)
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let (owned_depth_texture, depth_view, _owned_depth_sampler) =
            create_depth_texture(&device, &config, "depth_texture");

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        let uniforms = Uniforms::new();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let shader_code = include_str!("shader.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let camera = Camera::new((0.0, 1.5, 4.0), Deg(-90.0), Deg(-20.0));
        let camera_controller = CameraController::new(3.0, 0.002);

        let s = Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            owned_depth_texture,
            depth_view,
            _owned_depth_sampler,
            camera,
            camera_controller,
            model_angle: 0.0,
            model_rotation_speed: 0.5,
            window_focused: true,
            cursor_grab_toggled_on: true,
            last_update: Instant::now(),
        };
        s.apply_cursor_grab_state();
        s
    }

    fn apply_cursor_grab_state(&self) {
        let grab_mode = if self.cursor_grab_toggled_on && self.window_focused {
            if cfg!(target_arch = "wasm32") {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::Confined
            }
        } else {
            CursorGrabMode::None
        };
        if let Err(e) = self.window.set_cursor_grab(grab_mode) {
            log::warn!("Failed to set cursor grab mode: {:?}", e);
        }
        self.window
            .set_cursor_visible(grab_mode == CursorGrabMode::None);
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            let (new_depth_texture, new_depth_view, new_depth_sampler) =
                create_depth_texture(&self.device, &self.config, "depth_texture_resized");
            self.owned_depth_texture = new_depth_texture;
            self.depth_view = new_depth_view;
            self._owned_depth_sampler = new_depth_sampler;
        }
    }

    fn input_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } => {
                let mut consumed = self.window_focused
                    && self
                        .camera_controller
                        .process_keyboard(key_event.physical_key, key_event.state);
                if key_event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                    && key_event.state == ElementState::Pressed
                {
                    self.cursor_grab_toggled_on = !self.cursor_grab_toggled_on;
                    self.apply_cursor_grab_state();
                    consumed = true;
                }
                consumed
            }
            WindowEvent::Focused(focused) => {
                self.window_focused = *focused;
                self.apply_cursor_grab_state();
                true
            }
            _ => false,
        }
    }

    fn input_device_event(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                if self.cursor_grab_toggled_on && self.window_focused {
                    self.camera_controller
                        .process_mouse_motion(delta.0, delta.1, &mut self.camera);
                    return true;
                }
                false
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;

        if self.window_focused {
            self.camera_controller
                .update_camera_position(&mut self.camera, dt);
        }

        self.model_angle =
            (self.model_angle + self.model_rotation_speed * dt) % (2.0 * std::f32::consts::PI);

        let aspect_ratio = self.config.width as f32 / self.config.height as f32;
        self.uniforms
            .update_mvp(&self.camera, aspect_ratio, self.model_angle);
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output_frame = self.surface.get_current_texture()?;
        let surface_texture_view = output_frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output_frame.present();
        Ok(())
    }
}

impl ApplicationHandler for State {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        log::info!("Application resumed or started.");
        event_loop.set_control_flow(ControlFlow::Poll);
        self.window.request_redraw();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if window_id == self.window.id() {
            if !self.input_window_event(&event) {
                match event {
                    WindowEvent::CloseRequested => {
                        log::info!("WindowEvent::CloseRequested received.");
                        event_loop.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        log::info!("Window resized to: {:?}", physical_size);
                        self.resize(physical_size);
                    }
                    WindowEvent::RedrawRequested => {
                        self.update();
                        match self.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => {
                                log::warn!("Surface lost, resizing to reconfigure.");
                                self.resize(self.size);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("OutOfMemory error during render! Exiting.");
                                event_loop.exit();
                            }
                            Err(e) => log::error!("Error during render: {:?}", e),
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        self.input_device_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.window.request_redraw();
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        log::info!("Application exiting.");
    }
}

pub async fn run_app_core(window: Arc<Window>, event_loop: EventLoop<()>) {
    let mut state = State::new(window.clone()).await;

    match event_loop.run_app(&mut state) {
        Ok(_) => log::info!("Event loop finished cleanly."),
        Err(e) => log::error!("Event loop error: {:?}", e),
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_start_entry_point() {
    use std::sync::Arc;
    use winit::event_loop::EventLoop; 
    use winit::window::WindowAttributes;

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    match env_logger::try_init() {
        Ok(_) => log::info!("env_logger initialized for WASM"),
        Err(e) => web_sys::console::error_1(
            &format!("Failed to initialize env_logger for WASM: {:?}", e).into(),
        ),
    }
    log::info!("WASM module starting...");

    let event_loop = EventLoop::new().expect("Failed to create event loop for WASM");
    let window_attributes = WindowAttributes::default().with_title("WASM Cube App");
    #[allow(deprecated)] 
    let window = Arc::new(
        event_loop
            .create_window(window_attributes)
            .expect("Failed to build window for WASM"),
    );

    #[cfg(target_arch = "wasm32")]
    {
        let canvas = window
            .canvas()
            .expect("Window doesn't have a canvas for WASM");
        if canvas.id().is_empty() {
            canvas.set_id("wasm-canvas");
        }
        canvas.set_tab_index(0);

        let web_window = web_sys::window().unwrap();
        let document = web_window.document().unwrap();
        let body = document.body().unwrap();
        let style = canvas.style();
        style
            .set_property("display", "block")
            .expect("Failed to set canvas display style");
        style
            .set_property("width", "100vw")
            .expect("Failed to set canvas width");
        style
            .set_property("height", "100vh")
            .expect("Failed to set canvas height");

        if !body.contains(Some(&canvas)) {
            match body.append_child(&canvas) {
                Ok(_) => log::info!("Canvas appended to body."),
                Err(e) => web_sys::console::error_1(
                    &format!("Failed to append canvas to body: {:?}", e).into(),
                ),
            }
        } else {
            log::info!("Canvas was already in the body.");
        }
    }
    wasm_bindgen_futures::spawn_local(run_app_core(window, event_loop));
}
