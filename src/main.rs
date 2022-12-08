use byteorder::{LittleEndian, WriteBytesExt};
use clap::{Arg, Command};
use image::{ImageFormat, Rgb, RgbImage, RgbaImage};
use nalgebra::{DMatrix, Dynamic, VecStorage};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    borrow::Cow,
    fs::File,
    io::{BufReader, Write},
    num::{NonZeroU32, NonZeroU64},
    path::PathBuf,
    sync::mpsc,
    time::Instant,
};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Adapter, Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlendState,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, Device, DeviceDescriptor, Extent3d, FragmentState, ImageCopyBuffer,
    ImageCopyTexture, ImageDataLayout, Instance, Maintain, MapMode, MultisampleState, Operations,
    Origin3d, PipelineLayout, PipelineLayoutDescriptor, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, ShaderStages, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexBufferLayout, VertexState, VertexStepMode,
};

const SIZE: u32 = 512;
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
        let mut img = RgbImage::new(SIZE, SIZE);
        for yi in 0..SIZE {
            for xi in 0..SIZE {
                let x = 8.0 * (xi as f64 - (SIZE / 2) as f64) / SIZE as f64;
                let y = 8.0 * (yi as f64 - (SIZE / 2) as f64) / SIZE as f64;
                let t = 10.0 * ti as f64 / FRAMES_UNIT as f64;
                let mut tmp = DMatrix::from_iterator(
                    1,
                    11,
                    [
                        t,
                        1.0,
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
    for yi in 0..SIZE {
        for xi in 0..SIZE {
            let x = 8.0 * (xi as f64 - (SIZE / 2) as f64) / SIZE as f64;
            let y = 8.0 * (yi as f64 - (SIZE / 2) as f64) / SIZE as f64;
            let mut intermediates = Vec::new();
            let mut fw = DMatrix::from_iterator(
                1,
                11,
                [
                    t,
                    1.0,
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
            intermediates.push(fw.clone());
            for w in weights.iter() {
                fw *= w;
                intermediates.push(fw.clone());
                for v in fw.iter_mut() {
                    *v = v.tanh();
                }
            }
            //println!("{:?}", intermediates);
            let target_pixel = target.get_pixel(xi, yi);
            let target_vector = DMatrix::from_iterator(
                1,
                3,
                [
                    target_pixel.0[0] as f64 / 255.0,
                    target_pixel.0[1] as f64 / 255.0,
                    target_pixel.0[2] as f64 / 255.0,
                ],
            );
            //println!("{:?}", target_vector);
            let mut bk = fw - target_vector;
            for (fw, w) in intermediates.iter().zip(weights.iter_mut()).rev() {
                let fw_activated = fw.map(|v| v.tanh());
                let fw_prime = fw.map(|v| 1.0 - v.tanh().powf(2.0));
                //println!("{:?} {:?} {:?}", w.shape(), bk.shape(), fw.shape());
                let w_prime = bk.clone().transpose() * fw_activated;
                //println!("{:?} {:?} {:?}", w.shape(), bk.shape(), fw.shape());
                bk = fw_prime.zip_map(&(bk * w.transpose()), |a, b| a * b);
                //println!("w_prime: {:?} {:?}", w_prime.shape(), w_prime);
                w.zip_apply(&w_prime.transpose(), |a, b| *a -= alpha * b);
            }
        }
    }
}

struct GPUContext {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    shaders: ShaderModule,
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
}

impl GPUContext {
    async fn new(dims: &[usize]) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = Instance::new(Backends::PRIMARY);
        let adapter = instance
            .enumerate_adapters(Backends::PRIMARY)
            .next()
            .unwrap();
        println!("{:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await?;
        println!("{:?}", device);
        let shaders = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
        });
        let matrices_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
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
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(4),
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
            matrices_bind_group_layout,
            scalar_bind_group_layout,
        })
    }
}

impl ForwardPass {
    fn new(ctx: &GPUContext) -> Self {
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
            size: (4 * SIZE * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let size_extent = Extent3d {
            width: SIZE,
            height: SIZE,
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
        }
    }
}

fn sample_gpu(
    ctx: &GPUContext,
    pass: &ForwardPass,
    dims: &[usize],
    weights: &[DMatrix<f64>],
    frames: usize,
) -> Result<Vec<RgbaImage>, Box<dyn std::error::Error>> {
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
    let matrices_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &matrices_buffer_contents,
        usage: BufferUsages::STORAGE,
    });
    let matrices_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &ctx.matrices_bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(matrices_buffer.as_entire_buffer_binding()),
        }],
    });
    let mut images = Vec::new();
    for ti in 0..frames {
        let t = 10.0 * ti as f32 / FRAMES_UNIT as f32;
        let mut scalar_buffer_contents = Vec::new();
        scalar_buffer_contents.write(&t.to_le_bytes())?;
        let scalar_buffer = ctx.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &scalar_buffer_contents,
            usage: BufferUsages::UNIFORM,
        });
        let scalar_bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &ctx.scalar_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(scalar_buffer.as_entire_buffer_binding()),
            }],
        });
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
                        bytes_per_row: NonZeroU32::new(4 * SIZE),
                        rows_per_image: NonZeroU32::new(SIZE),
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
            let image = RgbaImage::from_raw(SIZE, SIZE, image_bytes).unwrap();
            images.push(image);
        }
        scalar_buffer.destroy();
    }

    Ok(images)
}

enum Backend {
    Cpu,
    Gpu,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new("perceptron-procgen")
        .arg(
            Arg::new("backend")
                .short('b')
                .value_parser(["cpu", "gpu"])
                .default_value("gpu")
                .global(true),
        )
        .arg(
            Arg::new("dimensions")
                .short('d')
                .default_value("[11, 10, 10, 3]")
                .global(true),
        )
        .subcommand(Command::new("forward"))
        .subcommand(
            Command::new("backward")
                .arg(Arg::new("target_image").short('t').required(true))
                .arg(Arg::new("weights").short('w')),
        );
    let command_help = command.render_help();
    let matches = command.get_matches();
    let backend = match matches.get_one::<String>("backend").map(|x| &**x) {
        Some("cpu") => Backend::Cpu,
        Some("gpu") => Backend::Gpu,
        _ => {
            return Ok(());
        }
    };
    let dims =
        serde_json::from_str::<Vec<usize>>(&**matches.get_one::<String>("dimensions").unwrap())?;

    match matches.subcommand() {
        Some(("forward", matches)) => {
            let ctx = futures_executor::block_on(GPUContext::new(&dims))?;
            let pass = ForwardPass::new(&ctx);
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
            let target_image_path =
                PathBuf::from(matches.get_one::<String>("target_image").unwrap());
            let file = File::open(&target_image_path)?;
            let reader = BufReader::new(file);
            let target_image =
                image::load(reader, ImageFormat::from_path(&target_image_path).unwrap())?;
            let target_image = target_image.to_rgb8();
            let ctx = futures_executor::block_on(GPUContext::new(&dims))?;
            let pass = ForwardPass::new(&ctx);
            for i in 0..500 {
                let pre = Instant::now();
                backprop_cpu(&mut weights, 0.0, 0.001, &target_image);
                let post = Instant::now();
                let duration = post.duration_since(pre);
                println!("backprop pass {}, {} seconds", i, duration.as_secs_f32());
                println!(
                    "{}",
                    serde_json::to_string(
                        &weights
                            .iter()
                            .map(|w| w.data.as_vec().clone())
                            .collect::<Vec<Vec<f64>>>()
                    )?
                );
                let imgs = sample_gpu(&ctx, &pass, &dims, &weights, 1)?;
                imgs[0].save(&format!("backprop_{:02}.png", i))?;
            }
        }
        _ => {
            println!("{}", command_help);
        }
    }

    Ok(())
}
