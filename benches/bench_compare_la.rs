use std::num::NonZeroU32;

#[cfg(not(target_arch = "wasm32"))]
use glassbench::*;

#[cfg(target_arch = "wasm32")]
use benchmark_simple::*;

use fast_image_resize::pixels::U8x2;
use fast_image_resize::{CpuExtensions, FilterType, Image, MulDiv, PixelType, ResizeAlg, Resizer};
use testing::PixelTestingExt;

mod utils;

pub fn bench_downscale_la(bench: &mut Bench) {
    let new_width = NonZeroU32::new(852).unwrap();
    let new_height = NonZeroU32::new(567).unwrap();

    let alg_names = ["Nearest", "Bilinear", "CatmullRom", "Lanczos3"];

    // fast_image_resize crate;
    let mut cpu_ext_and_name = vec![(CpuExtensions::None, "rust")];
    #[cfg(target_arch = "x86_64")]
    {
        cpu_ext_and_name.push((CpuExtensions::Sse4_1, "sse4.1"));
        cpu_ext_and_name.push((CpuExtensions::Avx2, "avx2"));
    }
    #[cfg(target_arch = "aarch64")]
    {
        cpu_ext_and_name.push((CpuExtensions::Neon, "neon"));
    }
    #[cfg(target_arch = "wasm32")]
    {
        cpu_ext_and_name.push((CpuExtensions::Wasm32, "wasm32"));
    }
    for (cpu_ext, ext_name) in cpu_ext_and_name {
        for alg_name in alg_names {
            let resize_alg = match alg_name {
                "Nearest" => ResizeAlg::Nearest,
                "Bilinear" => ResizeAlg::Convolution(FilterType::Bilinear),
                "CatmullRom" => ResizeAlg::Convolution(FilterType::CatmullRom),
                "Lanczos3" => ResizeAlg::Convolution(FilterType::Lanczos3),
                _ => return,
            };
            let src_image = U8x2::load_big_src_image();
            let src_view = src_image.view();
            let mut premultiplied_src_image =
                Image::new(src_image.width(), src_image.height(), src_view.pixel_type());
            let mut dst_image = Image::new(new_width, new_height, PixelType::U8x2);
            let mut dst_view = dst_image.view_mut();
            let mut mul_div = MulDiv::default();

            let mut fast_resizer = Resizer::new(resize_alg);

            unsafe {
                fast_resizer.reset_internal_buffers();
                fast_resizer.set_cpu_extensions(cpu_ext);
                mul_div.set_cpu_extensions(cpu_ext);
            }

            #[cfg(not(target_arch = "wasm32"))]
            bench.task(
                format!("fir {} - {}", ext_name, alg_name),
                |task| match resize_alg {
                    ResizeAlg::Nearest => {
                        task.iter(|| {
                            fast_resizer.resize(&src_view, &mut dst_view).unwrap();
                        });
                    }
                    _ => {
                        task.iter(|| {
                            let mut premultiplied_view = premultiplied_src_image.view_mut();
                            mul_div
                                .multiply_alpha(&src_view, &mut premultiplied_view)
                                .unwrap();
                            fast_resizer
                                .resize(&premultiplied_view.into(), &mut dst_view)
                                .unwrap();
                            mul_div.divide_alpha_inplace(&mut dst_view).unwrap();
                        });
                    }
                },
            );

            #[cfg(target_arch = "wasm32")]
            {
                let mut options = Options::default();
                options.iterations = 10;
                let res = match resize_alg {
                    ResizeAlg::Nearest => bench.run(&options, || {
                        fast_resizer.resize(&src_view, &mut dst_view).unwrap();
                    }),
                    _ => bench.run(&options, || {
                        let mut premultiplied_view = premultiplied_src_image.view_mut();
                        mul_div
                            .multiply_alpha(&src_view, &mut premultiplied_view)
                            .unwrap();
                        fast_resizer
                            .resize(&premultiplied_view.into(), &mut dst_view)
                            .unwrap();
                        mul_div.divide_alpha_inplace(&mut dst_view).unwrap();
                    }),
                };
                println!("fir {} - {}: {}", ext_name, alg_name, res);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    utils::print_md_table(bench);
}

#[cfg(not(target_arch = "wasm32"))]
bench_main!("Compare resize of LA image", bench_downscale_la,);

#[cfg(target_arch = "wasm32")]
pub fn main() {
    println!("Compare resize of LA image");
    let mut bench = Bench::new();
    bench_downscale_la(&mut bench);
}
