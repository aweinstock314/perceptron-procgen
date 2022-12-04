use image::{Rgb, RgbImage};
use nalgebra::DMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};

const SIZE: u32 = 512;

fn sample(seed: u64) -> RgbImage {
    let dims = vec![10, 10, 10, 3];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = vec![];
    for (i, j) in dims.iter().zip(dims[1..].iter()) {
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| rng.gen::<f64>()));
        weights.push(DMatrix::from_fn(*i, *j, |_, _| 2.0 * rng.gen::<f64>() - 1.0));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| (2.0 * rng.gen::<f64>() - 1.0) * (3.6 / (*i as f64 + *j as f64)).sqrt()));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| -1.0 * (*i as f64 + *j as f64)));
        //weights.push(DMatrix::from_fn(*i, *j, |_, _| 0.0));
    }

    let mut img = RgbImage::new(SIZE, SIZE);
    let (mut lo, mut hi) = (f64::INFINITY, -f64::INFINITY);
    for yi in 0..SIZE {
        for xi in 0..SIZE {
            let x = 8.0 * ((xi as f64 - (SIZE / 2) as f64)) / SIZE as f64;
            let y = 8.0 * ((yi as f64 - (SIZE / 2) as f64)) / SIZE as f64;
            let mut tmp = DMatrix::from_iterator(1, 10, [1.0, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y].into_iter());
            for w in weights.iter() {
                tmp *= w;
                for v in tmp.iter_mut() {
                    let epx = (*v).exp();
                    let emx = (-*v).exp();
                    *v = (epx - emx) / (epx + emx);
                }
            }
            for v in tmp.iter_mut() {
                if *v < lo {
                    lo = *v;
                }
                if *v > hi {
                    hi = *v;
                }
                *v = (*v + 1.0) / 2.0;
            }
            img.put_pixel(xi, yi, Rgb([(255.0 * tmp[0]) as u8, (255.0 * tmp[1]) as u8, (255.0 * tmp[2]) as u8]));
        }
    }
    println!("{} {}", lo, hi);
    img
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for seed in 0..20 {
        let img = sample(seed);
        img.save(&format!("tmp{:02}.png", seed))?;
    }
    Ok(())
}
