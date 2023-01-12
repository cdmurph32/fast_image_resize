#[cfg(not(target_arch = "wasm32"))]
use glassbench::*;

#[cfg(target_arch = "wasm32")]
use benchmark_simple::*;

use fast_image_resize::pixels::U8x3;
use fast_image_resize::{create_srgb_mapper, Image};
use testing::PixelTestingExt;

mod utils;

pub fn bench_color_mapper(bench: &mut Bench) {
    let src_image = U8x3::load_big_src_image();
    let mut dst_image = Image::new(
        src_image.width(),
        src_image.height(),
        src_image.pixel_type(),
    );
    let src_view = src_image.view();
    let mut dst_view = dst_image.view_mut();
    let mapper = create_srgb_mapper();
    #[cfg(not(target_arch = "wasm32"))]
    {
        bench.task("SRGB U8x3 => RGB U8x3", |task| {
            task.iter(|| {
                mapper.forward_map(&src_view, &mut dst_view).unwrap();
            })
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut options = Options::default();
        options.iterations = 10;
        let res = bench.run(&options, || {
            mapper.forward_map(&src_view, &mut dst_view).unwrap();
        });
        println!("SRGB U8x3 => RGB U8x3: {}", res);
    }
}

#[cfg(not(target_arch = "wasm32"))]
bench_main!("Bench color mappers", bench_color_mapper,);

#[cfg(target_arch = "wasm32")]
pub fn main() {
    println!("Bench color mappers");
    let mut bench = Bench::new();
    bench_color_mapper(&mut bench);
}
