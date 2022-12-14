use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::{value_parser, Arg, Command};
use image::{
    buffer::ConvertBuffer, ImageBuffer, ImageFormat, Luma, Rgb, RgbImage, Rgba, RgbaImage,
};
use nalgebra::{DMatrix, Dynamic, VecStorage, Vector3};
use rand::{distributions::Distribution, rngs::StdRng, Rng, SeedableRng};
use regex::{NoExpand, Regex};
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::File,
    io::{BufReader, Read, Write},
    num::{NonZeroU32, NonZeroU64},
    path::PathBuf,
    sync::mpsc,
    time::Instant,
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Extent3d, FragmentState, ImageCopyBuffer, ImageCopyTexture,
    ImageDataLayout, Instance, Maintain, MapMode, MultisampleState, Operations, Origin3d,
    PipelineLayout, PipelineLayoutDescriptor, PrimitiveState, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
    VertexBufferLayout, VertexState, VertexStepMode,
};

const VARIABLE_LEARNING_RATE: bool = false;
const FORWARD_SIZE: u32 = 512;
const FRAMES: usize = 1000;
const FRAMES_UNIT: usize = 100;

fn gen_weights(seed: u64, dims: &[usize]) -> Vec<DMatrix<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = vec![];
    for (i, j) in dims.iter().zip(dims[1..].iter()) {
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| rng.gen::<f64>()));
        weights.push(DMatrix::from_fn(*i, *j, |_, _| {
            2.0 * rng.gen::<f64>() - 1.0
        }));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| (2.0 * rng.gen::<f64>() - 1.0) * (3.6 / (*i as f64 + *j as f64)).sqrt()));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| -1.0 * (*i as f64 + *j as f64)));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| 0.0));
    }
    weights
}

fn sample(weights: &[DMatrix<f64>], frames: usize) -> Vec<RgbImage> {
    let mut imgs = Vec::new();
    for ti in 0..frames {
        let mut img = RgbImage::new(FORWARD_SIZE, FORWARD_SIZE);
        for yi in 0..FORWARD_SIZE {
            for xi in 0..FORWARD_SIZE {
                let x = 8.0 * (xi as f64 - (FORWARD_SIZE / 2) as f64) / FORWARD_SIZE as f64;
                let y = 8.0 * (yi as f64 - (FORWARD_SIZE / 2) as f64) / FORWARD_SIZE as f64;
                let t = 10.0 * ti as f64 / FRAMES_UNIT as f64;
                let mut tmp = DMatrix::from_iterator(
                    1,
                    11,
                    [
                        1.0,
                        t,
                        x,
                        y,
                        x * x,
                        x * y,
                        y * y,
                        x * x * x,
                        x * x * y,
                        x * y * y,
                        y * y * y,
                    ]
                    .into_iter(),
                );
                for w in weights.iter() {
                    tmp[0] = 1.0;
                    tmp *= w;
                    for v in tmp.iter_mut() {
                        let epx = (*v).exp();
                        let emx = (-*v).exp();
                        *v = (epx - emx) / (epx + emx);
                    }
                }
                for v in tmp.iter_mut() {
                    *v = (*v + 1.0) / 2.0;
                }
                img.put_pixel(
                    xi,
                    yi,
                    Rgb([
                        (255.0 * tmp[0]) as u8,
                        (255.0 * tmp[1]) as u8,
                        (255.0 * tmp[2]) as u8,
                    ]),
                );
            }
        }
        imgs.push(img);
        //println!("frame {}", ti);
    }
    imgs
}

fn backprop_cpu(weights: &mut [DMatrix<f64>], t: f64, alpha: f64, target: &RgbImage) {
    let mut weights_copy: Vec<DMatrix<f64>> = weights.iter().cloned().collect();
    let (width, height) = target.dimensions();
    for yi in 0..height {
        for xi in 0..width {
            let x = 8.0 * (xi as f64 - (width / 2) as f64) / width as f64;
            let y = 8.0 * (yi as f64 - (height / 2) as f64) / height as f64;
            let mut intermediates = Vec::new();
            let mut fw = DMatrix::from_iterator(
                1,
                11,
                [
                    1.0,
                    t,
                    x,
                    y,
                    x * x,
                    x * y,
                    y * y,
                    x * x * x,
                    x * x * y,
                    x * y * y,
                    y * y * y,
                ]
                .into_iter(),
            );
            intermediates.push(fw.map(|v| v.tanh()));
            for w in weights.iter() {
                fw[0] = 1.0;
                fw *= w;
                //intermediates.push(fw.clone());
                for v in fw.iter_mut() {
                    *v = v.tanh();
                }
                intermediates.push(fw.clone());
            }
            //println!("{:?}", intermediates);
            let target_pixel = target.get_pixel(xi, yi);
            let target_vector = DMatrix::from_iterator(
                1,
                3,
                [
                    2.0 * (target_pixel.0[0] as f64 / 255.0) - 1.0,
                    2.0 * (target_pixel.0[1] as f64 / 255.0) - 1.0,
                    2.0 * (target_pixel.0[2] as f64 / 255.0) - 1.0,
                ],
            );
            //println!("{:?}", target_vector);
            let mut bk = fw - target_vector;
            for (fw, w) in intermediates.iter().zip(weights_copy.iter_mut()).rev() {
                //let fw_activated = fw.map(|v| v.tanh());
                let fw_activated = fw;
                //let fw_prime = fw.map(|v| 1.0 - v.tanh().powf(2.0));
                let fw_prime = fw_activated.map(|v| 1.0 - v.powf(2.0));
                //println!("{:?} {:?} {:?}", w.shape(), bk.shape(), fw.shape());
                let w_prime = bk.clone().transpose() * fw_activated;
                //println!("{:?} {:?} {:?}", w.shape(), bk.shape(), fw.shape());
                bk = fw_prime.zip_map(&(bk * w.transpose()), |a, b| a * b);
                //println!("w_prime: {:?} {:?}", w_prime.shape(), w_prime);
                w.zip_apply(&w_prime.transpose(), |a, b| *a -= alpha * b);
            }
        }
    }
    for (w, wc) in weights.iter_mut().zip(weights_copy.into_iter()) {
        *w = wc;
    }
}

struct GPUContext {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    shaders: ShaderModule,
    shaders_parsed: naga::Module,
    matrices_bind_group_layout: BindGroupLayout,
    scalar_bind_group_layout: BindGroupLayout,
}

struct ForwardPass {
    pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    size_extent: Extent3d,
    texture: Texture,
    texture_view: TextureView,
    texture_buffer: Buffer,
    target_dims: (u32, u32),
}

struct BackwardPass {
    calc_norms_pipeline: ComputePipeline,
    backprop_pipeline: ComputePipeline,
    sum_delta_weights_pipeline: ComputePipeline,
    sum_weights_pipeline: ComputePipeline,
    forward_pipeline: ComputePipeline,
    forward_workgroup_size: u32,
    backprop_workgroup_size: u32,
    num_weights: usize,
    weights_offset: u64,
    weights_size: u64,
    weight_multiplicity: u64,
    output_weights: Buffer,
    output_weights_sum: Buffer,
    output_weights_copy: Buffer,
    dataset_points_bind_group_layout: BindGroupLayout,
    dataset_labels_bind_group_layout: BindGroupLayout,
    backprop_bind_group_layout: BindGroupLayout,
}

#[derive(Clone)]
struct Dataset {
    dimension: u32,
    count: u32,
    buffer_contents: Vec<u8>,
}

impl Dataset {
    fn new(dimension: u32) -> Dataset {
        let mut buffer_contents = Vec::new();
        buffer_contents.write(&dimension.to_le_bytes()).unwrap();
        buffer_contents.write(&0u32.to_le_bytes()).unwrap();
        Dataset {
            dimension,
            count: 0,
            buffer_contents,
        }
    }
    fn push(&mut self, point: &[f32]) {
        assert_eq!(point.len(), self.dimension as usize);
        self.count += 1;
        for i in point.iter() {
            self.buffer_contents.write(&i.to_le_bytes()).unwrap();
        }
    }
    fn to_buffer(&mut self, device: &Device) -> Buffer {
        (&mut self.buffer_contents[4..8])
            .write(&self.count.to_le_bytes())
            .unwrap();
        device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &self.buffer_contents,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
    }
    fn from_image_pixels(time: f32, image: &RgbImage) -> (Dataset, Dataset) {
        let mut data_buffer = Dataset::new(11);
        let mut label_buffer = Dataset::new(3);
        for y in 0..image.height() {
            let v = 8.0 * (2.0 * (y as f32 / image.height() as f32) - 1.0);
            for x in 0..image.width() {
                let u = 8.0 * (2.0 * (x as f32 / image.width() as f32) - 1.0);
                let pixel = image.get_pixel(x, y);
                let r: f32 = 2.0 * (pixel.0[0] as f32 / 255.0) - 1.0;
                let g: f32 = 2.0 * (pixel.0[1] as f32 / 255.0) - 1.0;
                let b: f32 = 2.0 * (pixel.0[2] as f32 / 255.0) - 1.0;
                //let (r, g, b) = (1.0f32, 1.0f32, 1.0f32);
                data_buffer.push(&[
                    1.0,
                    time,
                    u,
                    v,
                    u * u,
                    u * v,
                    v * v,
                    u * u * u,
                    u * u * v,
                    u * v * v,
                    v * v * v,
                ]);
                label_buffer.push(&[r, g, b]);
            }
        }
        (data_buffer, label_buffer)
    }
    fn blank_for_inference(time: f32, width: u32, height: u32) -> (Dataset, Dataset) {
        let mut data_buffer = Dataset::new(11);
        let mut label_buffer = Dataset::new(3);
        for y in 0..height {
            let v = 8.0 * ((y as f32 / height as f32) - 0.5);
            for x in 0..width {
                let u = 8.0 * ((x as f32 / width as f32) - 0.5);
                data_buffer.push(&[
                    1.0,
                    time,
                    u,
                    v,
                    u * u,
                    u * v,
                    v * v,
                    u * u * u,
                    u * u * v,
                    u * v * v,
                    v * v * v,
                ]);
                label_buffer.push(&[0.0, 0.0, 0.0]);
            }
        }
        (data_buffer, label_buffer)
    }
}

impl GPUContext {
    async fn new(dims: &[usize]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut shader_source = String::new();
        let mut shader_source_file = BufReader::new(File::open("src/shaders.wgsl")?);
        shader_source_file.read_to_string(&mut shader_source)?;
        let mut max_dim = *dims.iter().max().unwrap();
        max_dim += if max_dim % 4 != 0 { 4 - max_dim % 4 } else { 0 };
        let num_layers = dims.len();
        let replacements = [
            (
                "let NUM_LAYERS: u32 = \\d+u;",
                format!("let NUM_LAYERS: u32 = {}u;", num_layers),
            ),
            (
                "let MAX_DIM: u32 = \\d+u;",
                format!("let MAX_DIM: u32 = {}u;", max_dim),
            ),
            (
                "let MAX_DIM_QUARTER: u32 = \\d+u;",
                format!("let MAX_DIM_QUARTER: u32 = {}u;", max_dim / 4),
            ),
        ];
        for (pattern, replacement) in replacements.iter() {
            shader_source = Regex::new(pattern)
                .unwrap()
                .replace_all(&shader_source, NoExpand(replacement))
                .into_owned();
        }
        let instance = Instance::new(Backends::PRIMARY);
        let adapter = instance
            .enumerate_adapters(Backends::PRIMARY)
            .next()
            .unwrap();
        println!("{:?}", adapter.get_info());
        let mut device_descriptor = DeviceDescriptor::default();
        device_descriptor.limits.max_bind_groups = 5;
        let (device, queue) = adapter.request_device(&device_descriptor, None).await?;
        println!("{:?}", device);
        println!("{:?}", device.limits());
        let shaders_parsed = naga::front::wgsl::Parser::new().parse(&shader_source)?;
        let shaders = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            //source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
            source: ShaderSource::Wgsl(Cow::Owned(shader_source)),
        });
        let matrices_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4 * dims.len() as u64 + 4),
                    },
                    count: None,
                }],
            });
        let scalar_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT | ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(8),
                    },
                    count: None,
                }],
            });
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            shaders,
            shaders_parsed,
            matrices_bind_group_layout,
            scalar_bind_group_layout,
        })
    }
    fn matrices_bind_group(
        &self,
        dims: &[usize],
        weights: &[DMatrix<f64>],
    ) -> Result<(Buffer, BindGroup), Box<dyn std::error::Error>> {
        let mut matrices_buffer_contents = Vec::new();
        for dim in dims.iter() {
            matrices_buffer_contents.write_u32::<LittleEndian>(*dim as u32)?;
        }
        for w in weights.iter() {
            for row in w.row_iter() {
                for x in row.iter() {
                    matrices_buffer_contents.write(&(*x as f32).to_le_bytes())?;
                }
            }
        }
        let matrices_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &matrices_buffer_contents,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        let matrices_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.matrices_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(matrices_buffer.as_entire_buffer_binding()),
            }],
        });
        Ok((matrices_buffer, matrices_bind_group))
    }
    fn scalar_bind_group(
        &self,
        t: f32,
        alpha: f32,
    ) -> Result<(Buffer, BindGroup), Box<dyn std::error::Error>> {
        let mut scalar_buffer_contents = Vec::new();
        scalar_buffer_contents.write(&t.to_le_bytes())?;
        scalar_buffer_contents.write(&alpha.to_le_bytes())?;
        let scalar_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &scalar_buffer_contents,
            usage: BufferUsages::UNIFORM,
        });
        let scalar_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.scalar_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(scalar_buffer.as_entire_buffer_binding()),
            }],
        });
        Ok((scalar_buffer, scalar_bind_group))
    }
    fn workgroup_size_for_entry_point(&self, name: &str) -> Option<[u32; 3]> {
        self.shaders_parsed
            .entry_points
            .iter()
            .find(|ep| &ep.name == name)
            .map(|ep| ep.workgroup_size)
    }
}

impl ForwardPass {
    fn new(ctx: &GPUContext, target_dims: (u32, u32)) -> Self {
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &ctx.matrices_bind_group_layout,
                    &ctx.scalar_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let vertex_layout = VertexBufferLayout {
            array_stride: 0,
            step_mode: VertexStepMode::Vertex,
            attributes: &[],
        };
        let pipeline = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &ctx.shaders,
                    entry_point: &"vert_main",
                    buffers: &[vertex_layout],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &ctx.shaders,
                    entry_point: &"frag_main",
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: Some(BlendState::REPLACE),
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });
        let vertex_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 0,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        let texture_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4 * target_dims.0 as u64 * target_dims.1 as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let size_extent = Extent3d {
            width: target_dims.0,
            height: target_dims.1,
            depth_or_array_layers: 1,
        };
        let texture = ctx.device.create_texture(&TextureDescriptor {
            label: None,
            size: size_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        });
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        Self {
            pipeline,
            vertex_buffer,
            size_extent,
            texture,
            texture_view,
            texture_buffer,
            target_dims,
        }
    }
}

impl BackwardPass {
    fn new(
        ctx: &GPUContext,
        dims: &[usize],
        weight_multiplicity: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let weights_offset = 2 * 4 + 4 * dims.len() as u64;
        let dataset_points_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(12),
                        },
                        count: None,
                    }],
                });
        let dataset_labels_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(12),
                        },
                        count: None,
                    }],
                });
        let backprop_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        BindGroupLayoutEntry {
                            binding: 0,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: NonZeroU64::new(weights_offset + 4),
                            },
                            count: None,
                        },
                        BindGroupLayoutEntry {
                            binding: 1,
                            visibility: ShaderStages::COMPUTE,
                            ty: BindingType::Buffer {
                                ty: BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: NonZeroU64::new(4),
                            },
                            count: None,
                        },
                    ],
                });
        let backprop_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &ctx.matrices_bind_group_layout,
                        &ctx.scalar_bind_group_layout,
                        &dataset_points_bind_group_layout,
                        &dataset_labels_bind_group_layout,
                        &backprop_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let forward_pipeline_layout =
            ctx.device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &ctx.matrices_bind_group_layout,
                        &ctx.scalar_bind_group_layout,
                        &dataset_points_bind_group_layout,
                        &dataset_labels_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });
        let forward_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&forward_pipeline_layout),
                module: &ctx.shaders,
                entry_point: &"forward_gpu",
            });
        let forward_workgroup_size = ctx.workgroup_size_for_entry_point("forward_gpu").unwrap()[0];
        let backprop_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&backprop_pipeline_layout),
                module: &ctx.shaders,
                entry_point: &"backprop_gpu",
            });
        let backprop_workgroup_size =
            ctx.workgroup_size_for_entry_point("backprop_gpu").unwrap()[0];
        let sum_delta_weights_pipeline =
            ctx.device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&backprop_pipeline_layout),
                    module: &ctx.shaders,
                    entry_point: &"sum_delta_weights",
                });
        let sum_weights_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&backprop_pipeline_layout),
                module: &ctx.shaders,
                entry_point: &"sum_weights",
            });
        let calc_norms_pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&backprop_pipeline_layout),
                module: &ctx.shaders,
                entry_point: &"calc_norms",
            });
        let mut num_weights = 0;
        for (i, j) in dims.iter().zip(dims[1..].iter()) {
            num_weights += *i * *j;
        }
        let weights_size = weights_offset + weight_multiplicity as u64 * 4 * num_weights as u64;
        let output_weights = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: weights_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        let output_weights_sum = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4 * num_weights as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut buf = &mut output_weights.slice(0..8).get_mapped_range_mut()[..];
            buf.write_u32::<LittleEndian>(num_weights as u32)?;
            buf.write_u32::<LittleEndian>(weight_multiplicity as u32)?;
        }
        output_weights.unmap();
        let output_weights_copy = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4 * num_weights as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self {
            calc_norms_pipeline,
            backprop_pipeline,
            sum_delta_weights_pipeline,
            sum_weights_pipeline,
            forward_pipeline,
            num_weights,
            forward_workgroup_size,
            backprop_workgroup_size,
            weights_offset,
            weights_size,
            weight_multiplicity,
            output_weights,
            output_weights_sum,
            output_weights_copy,
            dataset_points_bind_group_layout,
            dataset_labels_bind_group_layout,
            backprop_bind_group_layout,
        })
    }
    fn dataset_points_bind_group(
        &self,
        ctx: &GPUContext,
        target_points: &mut Dataset,
    ) -> Result<(Buffer, BindGroup), Box<dyn std::error::Error>> {
        let target_points = target_points.to_buffer(&ctx.device);
        let dataset_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.dataset_points_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(target_points.as_entire_buffer_binding()),
            }],
        });
        Ok((target_points, dataset_bind_group))
    }
    fn dataset_labels_bind_group(
        &self,
        ctx: &GPUContext,
        target_labels: &mut Dataset,
    ) -> Result<(Buffer, BindGroup), Box<dyn std::error::Error>> {
        let target_labels = target_labels.to_buffer(&ctx.device);
        let dataset_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.dataset_labels_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(target_labels.as_entire_buffer_binding()),
            }],
        });
        Ok((target_labels, dataset_bind_group))
    }
    fn backprop_bind_group(
        &self,
        ctx: &GPUContext,
    ) -> Result<BindGroup, Box<dyn std::error::Error>> {
        let backprop_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.backprop_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(
                        self.output_weights.as_entire_buffer_binding(),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(
                        self.output_weights_sum.as_entire_buffer_binding(),
                    ),
                },
            ],
        });
        Ok(backprop_bind_group)
    }
}

fn sample_gpu(
    ctx: &GPUContext,
    pass: &ForwardPass,
    dims: &[usize],
    weights: &[DMatrix<f64>],
    frames: usize,
) -> Result<Vec<RgbaImage>, Box<dyn std::error::Error>> {
    let (_, matrices_bind_group) = ctx.matrices_bind_group(dims, weights)?;
    let mut images = Vec::new();
    for ti in 0..frames {
        let t = 10.0 * ti as f32 / FRAMES_UNIT as f32;
        let (scalar_buffer, scalar_bind_group) = ctx.scalar_bind_group(t, 0.0)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &pass.texture_view,
                    resolve_target: None,
                    ops: Operations::default(),
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&pass.pipeline);
            render_pass.set_bind_group(0, &matrices_bind_group, &[]);
            render_pass.set_bind_group(1, &scalar_bind_group, &[]);
            render_pass.set_vertex_buffer(0, pass.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
            drop(render_pass);
            encoder.copy_texture_to_buffer(
                ImageCopyTexture {
                    texture: &pass.texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                ImageCopyBuffer {
                    buffer: &pass.texture_buffer,
                    layout: ImageDataLayout {
                        offset: 0,
                        bytes_per_row: NonZeroU32::new(4 * pass.target_dims.0),
                        rows_per_image: NonZeroU32::new(pass.target_dims.1),
                    },
                },
                pass.size_extent,
            );
        }
        let submission = ctx.queue.submit([encoder.finish()]);
        let (img_tx, img_rx) = mpsc::channel();
        pass.texture_buffer
            .slice(..)
            .map_async(MapMode::Read, move |_| {
                let _ = img_tx.send(());
            });
        while !ctx
            .device
            .poll(Maintain::WaitForSubmissionIndex(submission))
        {}
        while let Ok(()) = img_rx.recv() {
            let image_bytes = Vec::from_iter(
                pass.texture_buffer
                    .slice(..)
                    .get_mapped_range()
                    .iter()
                    .copied(),
            );
            pass.texture_buffer.unmap();
            let image =
                RgbaImage::from_raw(pass.target_dims.0, pass.target_dims.1, image_bytes).unwrap();
            images.push(image);
        }
        scalar_buffer.destroy();
    }

    Ok(images)
}

fn forward_gpu(
    ctx: &GPUContext,
    pass: &BackwardPass,
    dims: &[usize],
    weights: &[DMatrix<f64>],
    num_points: usize,
    dataset_points_bind_group: &BindGroup,
    label_buffer: &Buffer,
    target_labels_bind_group: &BindGroup,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let output_dim = dims[dims.len() - 1];
    let output_size = (4 * output_dim * num_points) as u64;
    let (_, matrices_bind_group) = ctx.matrices_bind_group(dims, weights)?;
    let label_buffer_copy = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: output_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let (_, scalar_bind_group) = ctx.scalar_bind_group(0.0, 0.0)?;
    /*let mut target_labels = Dataset::new(output_dim as u32);
    let blank_point = vec![0.0; output_dim];
    for _ in 0..num_points {
        target_labels.push(&blank_point);
    }
    let (label_buffer, dataset_labels_bind_group) =
        pass.dataset_labels_bind_group(&ctx, &mut target_labels)?;*/
    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        compute_pass.set_bind_group(0, &matrices_bind_group, &[]);
        compute_pass.set_bind_group(1, &scalar_bind_group, &[]);
        compute_pass.set_bind_group(2, &dataset_points_bind_group, &[]);
        compute_pass.set_bind_group(3, &target_labels_bind_group, &[]);
        compute_pass.set_pipeline(&pass.forward_pipeline);
        let mut num_workgroups = num_points as u32 / pass.forward_workgroup_size;
        if num_points as u32 % pass.forward_workgroup_size != 0 {
            num_workgroups += 1;
        }
        compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        drop(compute_pass);
        encoder.copy_buffer_to_buffer(&label_buffer, 8, &label_buffer_copy, 0, output_size);
    }
    let submission = ctx.queue.submit([encoder.finish()]);
    let (result_tx, result_rx) = mpsc::channel();
    label_buffer_copy
        .slice(..)
        .map_async(MapMode::Read, move |_| {
            let _ = result_tx.send(());
        });
    while !ctx
        .device
        .poll(Maintain::WaitForSubmissionIndex(submission))
    {}
    let mut result = Vec::new();
    while let Ok(()) = result_rx.recv() {
        let result_bytes = Vec::from_iter(
            label_buffer_copy
                .slice(..)
                .get_mapped_range()
                .iter()
                .copied(),
        );
        label_buffer_copy.unmap();
        let mut cursor = std::io::Cursor::new(result_bytes);
        for _ in 0..num_points {
            let mut label = Vec::new();
            for _ in 0..output_dim {
                label.push(f32::from_bits(cursor.read_u32::<LittleEndian>()?));
            }
            result.push(label);
        }
    }
    Ok(result)
}

fn sample_gpu_compute(
    ctx: &GPUContext,
    pass: &BackwardPass,
    dims: &[usize],
    weights: &[DMatrix<f64>],
    width: u32,
    height: u32,
    frames: usize,
) -> Result<Vec<RgbaImage>, Box<dyn std::error::Error>> {
    let (_, matrices_bind_group) = ctx.matrices_bind_group(dims, weights)?;
    let mut images = Vec::new();
    let label_buffer_copy = ctx.device.create_buffer(&BufferDescriptor {
        label: None,
        size: 3 * 4 * width as u64 * width as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    for ti in 0..frames {
        let t = 10.0 * ti as f32 / FRAMES_UNIT as f32;
        let (scalar_buffer, scalar_bind_group) = ctx.scalar_bind_group(t, 0.0)?;
        let (mut target_points, mut target_labels) = Dataset::blank_for_inference(t, width, height);
        let (_, dataset_points_bind_group) =
            pass.dataset_points_bind_group(&ctx, &mut target_points)?;
        let (label_buffer, dataset_labels_bind_group) =
            pass.dataset_labels_bind_group(&ctx, &mut target_labels)?;
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            compute_pass.set_bind_group(0, &matrices_bind_group, &[]);
            compute_pass.set_bind_group(1, &scalar_bind_group, &[]);
            compute_pass.set_bind_group(2, &dataset_points_bind_group, &[]);
            compute_pass.set_bind_group(3, &dataset_labels_bind_group, &[]);
            compute_pass.set_pipeline(&pass.forward_pipeline);
            compute_pass.dispatch_workgroups((width * height) / pass.forward_workgroup_size, 1, 1);
            drop(compute_pass);
            encoder.copy_buffer_to_buffer(
                &label_buffer,
                8,
                &label_buffer_copy,
                0,
                3 * 4 * width as u64 * height as u64,
            );
        }
        let submission = ctx.queue.submit([encoder.finish()]);
        let (img_tx, img_rx) = mpsc::channel();
        label_buffer_copy
            .slice(..)
            .map_async(MapMode::Read, move |_| {
                let _ = img_tx.send(());
            });
        while !ctx
            .device
            .poll(Maintain::WaitForSubmissionIndex(submission))
        {}
        while let Ok(()) = img_rx.recv() {
            let image_bytes = Vec::from_iter(
                label_buffer_copy
                    .slice(..)
                    .get_mapped_range()
                    .iter()
                    .copied(),
            );
            label_buffer_copy.unmap();
            let mut image = RgbaImage::new(width, height);
            let mut cursor = std::io::Cursor::new(image_bytes);
            let f = |v: u32| -> u8 { (255.0 * (f32::from_bits(v) + 1.0) / 2.0) as u8 };
            for y in 0..height {
                for x in 0..width {
                    let pixel = Rgba([
                        f(cursor.read_u32::<LittleEndian>()?),
                        f(cursor.read_u32::<LittleEndian>()?),
                        f(cursor.read_u32::<LittleEndian>()?),
                        255,
                    ]);
                    image.put_pixel(x, y, pixel);
                }
            }
            images.push(image);
        }
        scalar_buffer.destroy();
    }

    Ok(images)
}

fn backprop_gpu(
    ctx: &GPUContext,
    pass: &BackwardPass,
    dims: &[usize],
    weights: &[DMatrix<f64>],
    weights_delta: &mut [DMatrix<f64>],
    t: f32,
    alpha: f32,
    epochs: usize,
    dataset_bind_groups: &[(u32, &BindGroup, &BindGroup)],
    backprop_bind_group: &BindGroup,
) -> Result<(), Box<dyn std::error::Error>> {
    assert!(epochs == 1 || dataset_bind_groups.len() == 1);
    let (matrices_buffer, matrices_bind_group) = ctx.matrices_bind_group(dims, weights)?;
    let (scalar_buffer, scalar_bind_group) = ctx.scalar_bind_group(t, alpha)?;
    let mut submission = None;
    ctx.device.start_capture();
    let num_datasets = dataset_bind_groups.len();
    for (i, (dataset_size, dataset_points_bind_group, dataset_labels_bind_group)) in
        dataset_bind_groups.iter().enumerate()
    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        {
            for _ in 0..epochs {
                if i == 0 {
                    encoder.clear_buffer(
                        &pass.output_weights,
                        pass.weights_offset,
                        NonZeroU64::new(pass.output_weights.size() - pass.weights_offset),
                    );
                }
                let mut compute_pass =
                    encoder.begin_compute_pass(&ComputePassDescriptor::default());
                compute_pass.set_bind_group(0, &matrices_bind_group, &[]);
                compute_pass.set_bind_group(1, &scalar_bind_group, &[]);
                compute_pass.set_bind_group(4, backprop_bind_group, &[]);
                {
                    compute_pass.set_bind_group(2, dataset_points_bind_group, &[]);
                    compute_pass.set_bind_group(3, dataset_labels_bind_group, &[]);
                    if i == 0 {
                        compute_pass.set_pipeline(&pass.calc_norms_pipeline);
                        compute_pass.dispatch_workgroups(1, 1, 1);
                    }
                    compute_pass.set_pipeline(&pass.backprop_pipeline);
                    let mut num_workgroups = dataset_size / pass.backprop_workgroup_size;
                    if dataset_size % pass.backprop_workgroup_size != 0 {
                        num_workgroups += 1;
                    }
                    compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
                }
                if i == num_datasets - 1 {
                    compute_pass.set_pipeline(&pass.sum_delta_weights_pipeline);
                    compute_pass.dispatch_workgroups(1, 1, 1);
                    if epochs > 1 {
                        compute_pass.set_pipeline(&pass.sum_weights_pipeline);
                        compute_pass.dispatch_workgroups(1, 1, 1);
                        drop(compute_pass);
                        encoder.copy_buffer_to_buffer(
                            &pass.output_weights_sum,
                            0,
                            &matrices_buffer,
                            4 * dims.len() as u64,
                            4 * pass.num_weights as u64,
                        );
                    }
                }
            }
            if i == num_datasets - 1 {
                encoder.copy_buffer_to_buffer(
                    &pass.output_weights,
                    pass.weights_offset,
                    &pass.output_weights_copy,
                    0,
                    4 * pass.num_weights as u64,
                );
            }
        }
        submission = Some(ctx.queue.submit([encoder.finish()]));
        while !ctx
            .device
            .poll(Maintain::WaitForSubmissionIndex(submission.unwrap()))
        {}
    }
    let (weights_tx, weights_rx) = mpsc::channel();
    pass.output_weights_copy
        .slice(..)
        .map_async(MapMode::Read, move |_| {
            let _ = weights_tx.send(());
        });
    while !ctx
        .device
        .poll(Maintain::WaitForSubmissionIndex(submission.unwrap()))
    {}
    ctx.device.stop_capture();
    while let Ok(()) = weights_rx.recv() {
        let weights_slice = pass.output_weights_copy.slice(..).get_mapped_range();
        let mut cursor = std::io::Cursor::new(weights_slice);
        for w in weights_delta.iter_mut() {
            for mut row in w.row_iter_mut() {
                for x in row.iter_mut() {
                    let mut buf = [0u8; 4];
                    cursor.read(&mut buf[..])?;
                    *x += f32::from_le_bytes(buf) as f64;
                }
            }
        }
        drop(cursor);
        pass.output_weights_copy.unmap();
    }
    scalar_buffer.destroy();
    Ok(())
}

enum Backend {
    Cpu,
    Gpu,
    GpuCompute,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new("perceptron-procgen")
        .arg(
            Arg::new("backend")
                .short('b')
                .value_parser(["cpu", "gpu", "gpu_compute"])
                .default_value("gpu")
                .global(true),
        )
        .arg(
            Arg::new("dimensions")
                .short('d')
                .default_value("[11, 10, 10, 3]")
                .global(true),
        )
        .arg(
            Arg::new("epochs")
                .short('e')
                .value_parser(value_parser!(usize))
                .default_value("4000")
                .global(true),
        )
        .arg(
            Arg::new("epochs_per_batch")
                .long("epochs-per-batch")
                .value_parser(value_parser!(usize))
                .default_value("8")
                .global(true),
        )
        .arg(
            Arg::new("learning_rate")
                .short('l')
                .value_parser(value_parser!(f64))
                .default_value("0.001")
                .global(true),
        )
        .subcommand(Command::new("forward"))
        .subcommand(
            Command::new("backward")
                .arg(Arg::new("target_image").short('t').required(true))
                .arg(Arg::new("weights").short('w')),
        )
        .subcommand(
            Command::new("infer_params").arg(Arg::new("target_image").short('t').required(true)),
        )
        .subcommand(
            Command::new("parse_mnist")
                .arg(Arg::new("labels_file").required(true))
                .arg(Arg::new("images_file").required(true))
                .arg(
                    Arg::new("save_images")
                        .value_parser(["true", "false"])
                        .default_value("false"),
                ),
        );
    let command_help = command.render_help();
    let matches = command.get_matches();
    let backend = match matches.get_one::<String>("backend").map(|x| &**x) {
        Some("cpu") => Backend::Cpu,
        Some("gpu") => Backend::Gpu,
        Some("gpu_compute") => Backend::GpuCompute,
        _ => {
            return Ok(());
        }
    };
    let mut dims =
        serde_json::from_str::<Vec<usize>>(&**matches.get_one::<String>("dimensions").unwrap())?;
    let epochs = matches.get_one::<usize>("epochs").unwrap();
    let epochs_per_batch = matches.get_one::<usize>("epochs_per_batch").unwrap();
    let mut alpha = *matches.get_one::<f64>("learning_rate").unwrap();

    match matches.subcommand() {
        Some(("forward", _matches)) => {
            let ctx = futures_executor::block_on(GPUContext::new(&dims))?;
            let pass = ForwardPass::new(&ctx, (FORWARD_SIZE, FORWARD_SIZE));
            let bk_pass = BackwardPass::new(&ctx, &dims, 1)?;
            for seed in 0..3 {
                let weights = gen_weights(seed, &dims);
                match backend {
                    Backend::Cpu => {
                        let pre = Instant::now();
                        let imgs = sample(&weights, FRAMES);
                        let post = Instant::now();
                        let duration = post.duration_since(pre);
                        println!(
                            "{} images in {} seconds",
                            imgs.len(),
                            duration.as_secs_f32()
                        );
                        for (i, img) in imgs.iter().enumerate() {
                            img.save(&format!("cpu{:02}_{:02}.png", seed, i))?;
                        }
                    }
                    Backend::Gpu => {
                        let pre = Instant::now();
                        let imgs = sample_gpu(&ctx, &pass, &dims, &weights, FRAMES)?;
                        let post = Instant::now();
                        let duration = post.duration_since(pre);
                        println!(
                            "{} images in {} seconds",
                            imgs.len(),
                            duration.as_secs_f32()
                        );
                        for (i, img) in imgs.iter().enumerate() {
                            img.save(&format!("gpu{:02}_{:02}.png", seed, i))?;
                        }
                    }
                    Backend::GpuCompute => {
                        let pre = Instant::now();
                        let imgs = sample_gpu_compute(
                            &ctx,
                            &bk_pass,
                            &dims,
                            &weights,
                            FORWARD_SIZE,
                            FORWARD_SIZE,
                            FRAMES,
                        )?;
                        let post = Instant::now();
                        let duration = post.duration_since(pre);
                        println!(
                            "{} images in {} seconds",
                            imgs.len(),
                            duration.as_secs_f32()
                        );
                        for (i, img) in imgs.iter().enumerate() {
                            img.save(&format!("gpu_compute_{:02}_{:02}.png", seed, i))?;
                        }
                    }
                }
            }
        }
        Some(("backward", matches)) => {
            let mut weights = if let Some(weights_json) = matches.get_one::<String>("weights") {
                let data = serde_json::from_str::<Vec<Vec<f64>>>(&weights_json)?;
                let mut weights = Vec::new();
                for (k, (i, j)) in dims.iter().zip(dims[1..].iter()).enumerate() {
                    let storage =
                        VecStorage::new(Dynamic::new(*i), Dynamic::new(*j), data[k].clone());
                    let w = DMatrix::from_vec_storage(storage);
                    weights.push(w);
                }
                weights
            } else {
                gen_weights(0, &dims)
            };
            let mut weights_delta = weights
                .iter()
                .map(|w| 0.0 * w)
                .collect::<Vec<DMatrix<f64>>>();
            let target_image_path =
                PathBuf::from(matches.get_one::<String>("target_image").unwrap());
            let file = File::open(&target_image_path)?;
            let reader = BufReader::new(file);
            let target_image =
                image::load(reader, ImageFormat::from_path(&target_image_path).unwrap())?;
            let target_image = target_image.to_rgb8();
            let ctx = futures_executor::block_on(GPUContext::new(&dims))?;
            let fw_pass = ForwardPass::new(&ctx, (FORWARD_SIZE, FORWARD_SIZE));
            let bk_pass = BackwardPass::new(&ctx, &dims, 8)?;
            let (mut target_points, mut target_labels) =
                Dataset::from_image_pixels(0.0, &target_image);
            let (_, dataset_points_bind_group) =
                bk_pass.dataset_points_bind_group(&ctx, &mut target_points)?;
            let (_, dataset_labels_bind_group) =
                bk_pass.dataset_labels_bind_group(&ctx, &mut target_labels)?;
            let backprop_bind_group = bk_pass.backprop_bind_group(&ctx)?;
            std::fs::create_dir_all("backprop_imgs")?;
            std::fs::create_dir_all("weights")?;
            let mut prev_error = std::f32::INFINITY;
            for i in 0..epochs / epochs_per_batch {
                let pre = Instant::now();
                match backend {
                    Backend::Cpu => {
                        for _ in 0..*epochs_per_batch {
                            backprop_cpu(
                                &mut weights,
                                0.0,
                                alpha
                                    / ((target_image.width() * target_image.height()) as f64)
                                        .sqrt(),
                                &target_image,
                            );
                        }
                    }
                    Backend::Gpu | Backend::GpuCompute => {
                        backprop_gpu(
                            &ctx,
                            &bk_pass,
                            &dims,
                            &weights,
                            &mut weights_delta,
                            0.0,
                            (alpha / ((target_image.width() * target_image.height()) as f64).sqrt())
                                as f32,
                            *epochs_per_batch,
                            &[(
                                target_points.count,
                                &dataset_points_bind_group,
                                &dataset_labels_bind_group,
                            )],
                            &backprop_bind_group,
                        )?;
                        for (w, d) in weights.iter_mut().zip(weights_delta.iter()) {
                            *w += d;
                        }
                    }
                }
                let post = Instant::now();
                let duration = post.duration_since(pre);
                if let Ok(mut f) = File::create(format!("weights/checkpoint_{:05}.json", i)) {
                    let serialized_weights = serde_json::to_string(
                        &weights
                            .iter()
                            .map(|w| w.data.as_vec().clone())
                            .collect::<Vec<Vec<f64>>>(),
                    )?;
                    writeln!(f, "{}", serialized_weights)?;
                }
                let imgs = sample_gpu_compute(
                    &ctx,
                    &bk_pass,
                    &dims,
                    &weights,
                    target_image.width(),
                    target_image.height(),
                    1,
                )?;
                let mut error = 0.0;
                for yi in 0..target_image.height() {
                    for xi in 0..target_image.width() {
                        let a = imgs[0].get_pixel(xi, yi);
                        let a = 2.0
                            * (Vector3::new(a.0[0] as f32, a.0[1] as f32, a.0[2] as f32) / 255.0)
                                .add_scalar(-1.0);
                        let b = target_image.get_pixel(xi, yi);
                        let b = 2.0
                            * (Vector3::new(b.0[0] as f32, b.0[1] as f32, b.0[2] as f32) / 255.0)
                                .add_scalar(-1.0);
                        error += (b - &a).norm();
                    }
                }
                println!(
                    "backprop pass {}, {} seconds, {} error, {} alpha",
                    i,
                    duration.as_secs_f32(),
                    error,
                    alpha
                );
                if VARIABLE_LEARNING_RATE {
                    if error > 1.02 * prev_error {
                        alpha *= 0.99;
                        prev_error = error;
                    }
                    prev_error = prev_error.min(error);
                }
                let imgs = match backend {
                    Backend::Cpu => sample(&weights, 1)
                        .into_iter()
                        .map(|i| i.convert())
                        .collect(),
                    Backend::Gpu => sample_gpu(&ctx, &fw_pass, &dims, &weights, 1)?,
                    Backend::GpuCompute => sample_gpu_compute(
                        &ctx,
                        &bk_pass,
                        &dims,
                        &weights,
                        FORWARD_SIZE,
                        FORWARD_SIZE,
                        1,
                    )?,
                };
                imgs[0].save(&format!("backprop_imgs/backprop_{:05}.png", i))?;
            }
        }
        Some(("infer_params", matches)) => {
            let weights = gen_weights(0, &dims);
            let weights_delta = weights
                .iter()
                .map(|w| 0.0 * w)
                .collect::<Vec<DMatrix<f64>>>();
            let target_image_path =
                PathBuf::from(matches.get_one::<String>("target_image").unwrap());
            let file = File::open(&target_image_path)?;
            let reader = BufReader::new(file);
            let target_image =
                image::load(reader, ImageFormat::from_path(&target_image_path).unwrap())?;
            let target_image = target_image.to_rgb8();
            let ctx = futures_executor::block_on(GPUContext::new(&dims))?;
            let mut parameters = Vec::new();
            for target_size in [16, 32, 64, 128, 256, 512].iter() {
                for epochs_per_batch in [1, 2, 4, 8, 16].iter() {
                    parameters.push((*target_size, *epochs_per_batch, 8));
                }
            }
            for weight_multiplicity in [/*1, 2,*/ 4, 8, 16, 24, 32, 40, 48, 56, 64].iter() {
                parameters.push((64, 16, *weight_multiplicity));
            }
            for (target_size, epochs_per_batch, weight_multiplicity) in parameters.iter() {
                let target_image = image::imageops::resize(
                    &target_image,
                    *target_size,
                    *target_size,
                    image::imageops::FilterType::Triangle,
                );
                let (mut target_points, mut target_labels) =
                    Dataset::from_image_pixels(0.0, &target_image);
                let bk_pass = BackwardPass::new(&ctx, &dims, *weight_multiplicity)?;
                let (_, dataset_points_bind_group) =
                    bk_pass.dataset_points_bind_group(&ctx, &mut target_points)?;
                let (_, dataset_labels_bind_group) =
                    bk_pass.dataset_labels_bind_group(&ctx, &mut target_labels)?;
                let backprop_bind_group = bk_pass.backprop_bind_group(&ctx)?;
                let mut weights = weights.clone();
                let mut weights_delta = weights_delta.clone();
                let pre = Instant::now();
                backprop_gpu(
                    &ctx,
                    &bk_pass,
                    &dims,
                    &weights,
                    &mut weights_delta,
                    0.0,
                    0.001,
                    *epochs_per_batch,
                    &[(
                        target_points.count,
                        &dataset_points_bind_group,
                        &dataset_labels_bind_group,
                    )],
                    &backprop_bind_group,
                )?;
                for (w, d) in weights.iter_mut().zip(weights_delta.iter()) {
                    *w += d;
                }
                let post = Instant::now();
                let duration = post.duration_since(pre);
                println!(
                    "backprop pass {}x{}, {} epb {} wm: {} seconds ({} per epoch)",
                    target_size,
                    target_size,
                    epochs_per_batch,
                    weight_multiplicity,
                    duration.as_secs_f32(),
                    duration.as_secs_f32() / *epochs_per_batch as f32
                );
            }
        }
        Some(("parse_mnist", matches)) => {
            let mut labels_file = BufReader::new(File::open(PathBuf::from(
                matches.get_one::<String>("labels_file").unwrap(),
            ))?);
            let mut images_file = BufReader::new(File::open(PathBuf::from(
                matches.get_one::<String>("images_file").unwrap(),
            ))?);
            let save_images =
                matches.get_one::<String>("save_images").map(|x| &**x) == Some("true");
            let magic = labels_file.read_u32::<BigEndian>()?;
            let num_labels = labels_file.read_u32::<BigEndian>()? as usize;
            assert_eq!(magic, 0x00000801, "MNIST labels magic mismatch");
            let magic = images_file.read_u32::<BigEndian>()?;
            assert_eq!(magic, 0x00000803, "MNIST images magic mismatch");
            let num_images = images_file.read_u32::<BigEndian>()? as usize;
            assert_eq!(
                num_labels, num_images,
                "Inconsistant amount of training points between label and image files."
            );
            let width = images_file.read_u32::<BigEndian>()?;
            let height = images_file.read_u32::<BigEndian>()?;
            const BATCH_SIZE_CAP: usize = 9000;
            let ctx = futures_executor::block_on(GPUContext::new(&dims))?;
            let mut batch_size = (ctx.device.limits().max_storage_buffer_binding_size as usize - 8)
                / (4 * width as usize * height as usize);
            batch_size = batch_size.min(BATCH_SIZE_CAP);
            batch_size += if batch_size % 16 != 0 {
                16 - batch_size % 16
            } else {
                0
            };
            let num_batches =
                (num_images / batch_size) + if (num_images % batch_size) != 0 { 1 } else { 0 };
            //assert_eq!(num_batches * batch_size, num_images);
            let mut target_labels = vec![Dataset::new(10); num_batches];
            let mut labels: BTreeMap<u32, u8> = BTreeMap::new();
            let mut label_vectors: BTreeMap<usize, [f32; 10]> = BTreeMap::new();
            for i in 0..num_labels {
                let value = labels_file.read_u8()?;
                labels.insert(i as u32, value);
                let mut point = [-1.0; 10];
                point[value as usize % 10] = 1.0;
                target_labels[i % num_batches].push(&point);
                label_vectors.insert(i, point);
            }
            let mut labels_json = File::create("mnist_labels.json")?;
            write!(labels_json, "{}", serde_json::to_string(&labels)?)?;
            if save_images {
                std::fs::create_dir_all("mnist_imgs")?;
            }
            let mut target_points = vec![Dataset::new(width * height); num_batches];
            let mut point_vectors: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
            for i in 0..num_images {
                let mut bytes = Vec::new();
                let mut point = Vec::new();
                for _ in 0..width * height {
                    let value = 255 - images_file.read_u8()?;
                    bytes.push(value);
                    point.push(value as f32 / 255.0);
                }
                if save_images {
                    let image =
                        ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width, height, bytes).unwrap();
                    image.save(format!("mnist_imgs/{:05}.png", i))?;
                }
                target_points[i % num_batches].push(&point);
                point_vectors.insert(i, point);
            }
            dims[0] = width as usize * height as usize;
            let dims_len = dims.len();
            dims[dims_len - 1] = 10;
            let mut weights = if let Some(weights_json) = matches.get_one::<String>("weights") {
                let data = serde_json::from_str::<Vec<Vec<f64>>>(&weights_json)?;
                let mut weights = Vec::new();
                for (k, (i, j)) in dims.iter().zip(dims[1..].iter()).enumerate() {
                    let storage =
                        VecStorage::new(Dynamic::new(*i), Dynamic::new(*j), data[k].clone());
                    let w = DMatrix::from_vec_storage(storage);
                    weights.push(w);
                }
                weights
            } else {
                gen_weights(0, &dims)
            };
            let bk_pass = BackwardPass::new(&ctx, &dims, 8)?;
            let mut dataset_bind_groups = Vec::new();
            for j in 0..num_batches {
                let (_, dataset_points_bind_group) =
                    bk_pass.dataset_points_bind_group(&ctx, &mut target_points[j])?;
                let (_, dataset_labels_bind_group) =
                    bk_pass.dataset_labels_bind_group(&ctx, &mut target_labels[j])?;
                dataset_bind_groups.push((
                    target_points[j].count,
                    dataset_points_bind_group,
                    dataset_labels_bind_group,
                ));
            }
            let (output_labels_buffer, output_labels_bind_group) =
                bk_pass.dataset_labels_bind_group(&ctx, &mut target_labels[0])?;
            let dataset_bind_group_refs = dataset_bind_groups
                .iter()
                .map(|(i, x, y)| (*i, x, y))
                .collect::<Vec<_>>();
            let backprop_bind_group = bk_pass.backprop_bind_group(&ctx)?;
            let mut rng = StdRng::seed_from_u64(0);
            let uniform = rand::distributions::Uniform::from(0..num_images);
            std::fs::create_dir_all("mnist_weights")?;
            for i in 0..*epochs / *epochs_per_batch {
                let mut weights_delta = weights
                    .iter()
                    .map(|w| 0.0 * w)
                    .collect::<Vec<DMatrix<f64>>>();
                const STOCHASTIC_GRADIENT_DESCENT: bool = false;
                if STOCHASTIC_GRADIENT_DESCENT {
                    const SGD_BATCH_SIZE: usize = 512;
                    let mut target_points = Dataset::new(784);
                    let mut target_labels = Dataset::new(10);
                    for _ in 0..SGD_BATCH_SIZE {
                        let index = uniform.sample(&mut rng);
                        target_points.push(&point_vectors[&index]);
                        target_labels.push(&label_vectors[&index]);
                    }
                    let (_, dataset_points_bind_group) =
                        bk_pass.dataset_points_bind_group(&ctx, &mut target_points)?;
                    let (_, dataset_labels_bind_group) =
                        bk_pass.dataset_labels_bind_group(&ctx, &mut target_labels)?;
                    let pre = Instant::now();
                    for d in weights_delta.iter_mut() {
                        *d *= 0.0;
                    }
                    backprop_gpu(
                        &ctx,
                        &bk_pass,
                        &dims,
                        &weights,
                        &mut weights_delta,
                        0.0,
                        alpha as f32,
                        *epochs_per_batch,
                        //&[dataset_bind_group_refs[j]],
                        &[(
                            target_points.count,
                            &dataset_points_bind_group,
                            &dataset_labels_bind_group,
                        )],
                        &backprop_bind_group,
                    )?;
                    for (w, d) in weights.iter_mut().zip(weights_delta.iter()) {
                        *w += d;
                    }
                    let post = Instant::now();
                    let duration = post.duration_since(pre);
                    println!("backprop pass {}: {} seconds", i, duration.as_secs_f32());
                } else {
                    const COPY_WEIGHTS_EVERY_BATCH: bool = false;
                    let total_pre = Instant::now();
                    for d in weights_delta.iter_mut() {
                        *d *= 0.0;
                    }
                    let jmax = if COPY_WEIGHTS_EVERY_BATCH {
                        num_batches
                    } else {
                        1
                    };
                    for j in 0..jmax {
                        let dbgr_tmp = [dataset_bind_group_refs[j]];
                        let dbgr = if COPY_WEIGHTS_EVERY_BATCH {
                            &dbgr_tmp[..]
                        } else {
                            &dataset_bind_group_refs[..]
                        };
                        let pre = Instant::now();
                        backprop_gpu(
                            &ctx,
                            &bk_pass,
                            &dims,
                            &weights,
                            &mut weights_delta,
                            0.0,
                            alpha as f32,
                            *epochs_per_batch,
                            dbgr,
                            &backprop_bind_group,
                        )?;
                        let post = Instant::now();
                        let duration = post.duration_since(pre);
                        println!(
                            "backprop pass {}, {}: {} seconds",
                            i,
                            j,
                            duration.as_secs_f32()
                        );
                    }
                    for (w, d) in weights.iter_mut().zip(weights_delta.iter()) {
                        *w += d;
                    }
                    let total_post = Instant::now();
                    let total_duration = total_post.duration_since(total_pre);
                    println!(
                        "backprop pass {}: {} seconds",
                        i,
                        total_duration.as_secs_f32()
                    );
                }
                if let Ok(mut f) = File::create(format!("mnist_weights/checkpoint_{:05}.json", i)) {
                    let serialized_weights = serde_json::to_string(
                        &weights
                            .iter()
                            .map(|w| w.data.as_vec().clone())
                            .collect::<Vec<Vec<f64>>>(),
                    )?;
                    writeln!(f, "{}", serialized_weights)?;
                }
                let mut error = 0.0;
                let mut batch_offset = 0;
                let mut f = File::create(format!("mnist_output_{:05}.json", i))?;
                for j in 0..num_batches {
                    let pre = Instant::now();
                    let generated_labels = forward_gpu(
                        &ctx,
                        &bk_pass,
                        &dims,
                        &weights,
                        dataset_bind_groups[j].0 as usize,
                        &dataset_bind_groups[j].1,
                        &output_labels_buffer,
                        &output_labels_bind_group,
                    )?;
                    let post = Instant::now();
                    let duration = post.duration_since(pre);
                    println!(
                        "forward pass {}, {}: {} points, {} seconds",
                        i,
                        j,
                        target_points[j].count,
                        duration.as_secs_f32()
                    );
                    for (k, generated_label) in generated_labels.iter().enumerate() {
                        writeln!(f, "{}: {:?}", batch_offset + k, generated_label)?;
                        let mut target_label = [-1.0; 10];
                        target_label[labels[&((batch_offset + k) as u32)] as usize % 10] = 1.0f32;
                        let target_label =
                            DMatrix::from_iterator(1, 10, target_label.iter().copied());
                        let generated_label =
                            DMatrix::from_iterator(1, 10, generated_label.iter().copied());
                        error += (generated_label - &target_label).norm_squared();
                    }
                    batch_offset += generated_labels.len();
                }
                println!("error {}: {}", i, error);
            }
        }
        _ => {
            println!("{}", command_help);
        }
    }

    Ok(())
}
