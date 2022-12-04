use image::{Rgb, RgbImage};
use nalgebra::DMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};

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

fn sample(seed: u64) -> Vec<RgbImage> {
    let dims = vec![11, 10, 10, 3];
    let weights = gen_weights(seed, &dims);

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
        println!("{} {}", seed, ti);
    }
    imgs
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for seed in 0..3 {
        let imgs = sample(seed);
        for (i, img) in imgs.iter().enumerate() {
            img.save(&format!("tmp{:02}_{:02}.png", seed, i))?;
        }
    }
    Ok(())
}
