use image::{Rgb, RgbImage, RgbaImage};
use nalgebra::DMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{borrow::Cow, num::NonZeroU32, sync::mpsc};
use wgpu::{Instance, Backends, DeviceDescriptor, ShaderModuleDescriptor, ShaderSource, VertexState, VertexBufferLayout, VertexStepMode, PrimitiveState, FragmentState, ColorWrites, ColorTargetState, PipelineLayout, RenderPipelineDescriptor, MultisampleState, TextureFormat, BlendState, PipelineLayoutDescriptor, TextureDescriptor, Extent3d, TextureUsages, TextureDimension
, Operations, RenderPassColorAttachment, CommandEncoderDescriptor, TextureViewDescriptor, RenderPassDescriptor, BufferDescriptor, BufferUsages, ImageDataLayout, ImageCopyTexture, ImageCopyBuffer, Origin3d, TextureAspect, Maintain, MapMode,
};

const SIZE: u32 = 512;
const FRAMES: usize = 100;

fn gen_weights(seed: u64, dims: &[usize]) -> Vec<DMatrix<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = vec![];
    for (i, j) in dims.iter().zip(dims[1..].iter()) {
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| rng.gen::<f64>()));
        weights.push(DMatrix::from_fn(*i, *j, |_, _| 2.0 * rng.gen::<f64>() - 1.0));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| (2.0 * rng.gen::<f64>() - 1.0) * (3.6 / (*i as f64 + *j as f64)).sqrt()));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| -1.0 * (*i as f64 + *j as f64)));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| 0.0));
    }
    weights
}

fn sample(weights: &[DMatrix<f64>]) -> Vec<RgbImage> {
    let mut imgs = Vec::new();
    for ti in 0..FRAMES {
        let mut img = RgbImage::new(SIZE, SIZE);
        for yi in 0..SIZE {
            for xi in 0..SIZE {
                let x = 8.0 * ((xi as f64 - (SIZE / 2) as f64)) / SIZE as f64;
                let y = 8.0 * ((yi as f64 - (SIZE / 2) as f64)) / SIZE as f64;
                let t = 10.0 * ti as f64 / FRAMES as f64;
                let mut tmp = DMatrix::from_iterator(1, 11, [t, 1.0, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y].into_iter());
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
                img.put_pixel(xi, yi, Rgb([(255.0 * tmp[0]) as u8, (255.0 * tmp[1]) as u8, (255.0 * tmp[2]) as u8]));
            }
        }
        imgs.push(img);
        println!("frame {}", ti);
    }
    imgs
}

async fn sample_gpu(weights: &[DMatrix<f64>]) -> Result<Vec<RgbaImage>, Box<dyn std::error::Error>> {
    let instance = Instance::new(Backends::PRIMARY);
    let adapter = instance.enumerate_adapters(Backends::PRIMARY).next().unwrap();
    println!("{:?}", adapter.get_info());
    let (device, queue) = adapter.request_device(&DeviceDescriptor::default(), None).await?;
    println!("{:?}", device);
    let shaders = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
    });
    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor::default());
    let vertex_layout = VertexBufferLayout {
        array_stride: 0,
        step_mode: VertexStepMode::Vertex,
        attributes: &[],
    };
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &shaders,
            entry_point: &"vert_main",
            buffers: &[vertex_layout],
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shaders,
            entry_point: &"frag_main",
            targets: &[
                Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })
            ],
        }),
        multiview: None,
    });
    let vertex_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: 0,
        usage: BufferUsages::VERTEX,
        mapped_at_creation: false,
    });
    let texture_buffer = device.create_buffer(&BufferDescriptor {
        label: None,
        size: (4 * SIZE * SIZE) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let size_extent = Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1, };
    let texture = device.create_texture(&TextureDescriptor {
        label: None,
        size: size_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
    });
    let texture_view = texture.create_view(&TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: Operations::default(),
                }),
            ],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.draw(0..6, 0..1);
        drop(render_pass);
        encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            ImageCopyBuffer {
                buffer: &texture_buffer, 
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(4 * SIZE),
                    rows_per_image: NonZeroU32::new(SIZE),
                },
            },
            size_extent
        );
    }
    let submission = queue.submit([encoder.finish()]);
    let (img_tx, img_rx) = mpsc::channel();
    texture_buffer.slice(..).map_async(MapMode::Read, move |_| {
        let _ = img_tx.send(());
    });
    while !device.poll(Maintain::WaitForSubmissionIndex(submission)) {}
    let mut images = Vec::new();
    while let Ok(()) = img_rx.recv() {
        let image_bytes = Vec::from_iter(texture_buffer.slice(..).get_mapped_range().iter().copied());
        println!("{:?}", &image_bytes[0..50]);
        let image = RgbaImage::from_raw(SIZE, SIZE, image_bytes).unwrap();
        images.push(image);
    }

    Ok(images)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dims = vec![11, 10, 10, 3];

    for seed in 0..3 {
        let weights = gen_weights(seed, &dims);
        //let imgs = sample(&weights);
        let imgs = futures_executor::block_on(sample_gpu(&weights))?;
        println!("imgs.len: {}", imgs.len());
        for (i, img) in imgs.iter().enumerate() {
            img.save(&format!("tmp{:02}_{:02}.png", seed, i))?;
        }
    }
    Ok(())
}
