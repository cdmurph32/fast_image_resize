use std::num::NonZeroU32;

#[cfg(not(target_arch = "wasm32"))]
use glassbench::*;
#[cfg(not(target_arch = "wasm32"))]
use std::thread::sleep;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(target_arch = "wasm32")]
use benchmark_simple::*;

use image::imageops;
use resize::Pixel::Gray16;
use rgb::alt::Gray;
use rgb::FromSlice;

use fast_image_resize::pixels::U16;
use fast_image_resize::Image;
use fast_image_resize::{CpuExtensions, FilterType, ResizeAlg, Resizer};
use testing::PixelTestingExt;

mod utils;

pub fn bench_downscale_l16(bench: &mut Bench) {
    let src_image = U16::load_big_image().to_luma16();
    let new_width = NonZeroU32::new(852).unwrap();
    let new_height = NonZeroU32::new(567).unwrap();

    let alg_names = ["Nearest", "Bilinear", "CatmullRom", "Lanczos3"];

    // image crate
    // https://crates.io/crates/image
    for alg_name in alg_names {
        let filter = match alg_name {
            "Nearest" => imageops::Nearest,
            "Bilinear" => imageops::Triangle,
            "CatmullRom" => imageops::CatmullRom,
            "Lanczos3" => imageops::Lanczos3,
            _ => continue,
        };
        #[cfg(not(target_arch = "wasm32"))]
        bench.task(format!("image - {}", alg_name), |task| {
            task.iter(|| {
                imageops::resize(&src_image, new_width.get(), new_height.get(), filter);
            })
        });
        #[cfg(target_arch = "wasm32")]
        {
            let mut options = Options::default();
            options.iterations = 10;
            let res = bench.run(&options, || {
                imageops::resize(&src_image, new_width.get(), new_height.get(), filter);
            });
            println!("image - {}: {}", alg_name, res);
        }
    }

    // resize crate
    // https://crates.io/crates/resize
    for alg_name in alg_names {
        let resize_src_image = src_image.as_raw().as_gray();
        let mut dst = vec![Gray(0u16); (new_width.get() * new_height.get()) as usize];
        #[cfg(not(target_arch = "wasm32"))]
        bench.task(format!("resize - {}", alg_name), |task| {
            let filter = match alg_name {
                "Nearest" => {
                    // resizer doesn't support "nearest" algorithm
                    task.iter(|| sleep(Duration::new(0, 1)));
                    return;
                }
                "Bilinear" => resize::Type::Triangle,
                "CatmullRom" => resize::Type::Catrom,
                "Lanczos3" => resize::Type::Lanczos3,
                _ => return,
            };
            let mut resize = resize::new(
                src_image.width() as usize,
                src_image.height() as usize,
                new_width.get() as usize,
                new_height.get() as usize,
                Gray16,
                filter,
            )
            .unwrap();
            task.iter(|| {
                resize.resize(resize_src_image, &mut dst).unwrap();
            })
        });

        #[cfg(target_arch = "wasm32")]
        {
            let filter = match alg_name {
                "Nearest" => {
                    continue;
                }
                "Bilinear" => resize::Type::Triangle,
                "CatmullRom" => resize::Type::Catrom,
                "Lanczos3" => resize::Type::Lanczos3,
                _ => return,
            };
            let mut resize = resize::new(
                src_image.width() as usize,
                src_image.height() as usize,
                new_width.get() as usize,
                new_height.get() as usize,
                Gray16,
                filter,
            )
            .unwrap();
            let mut options = Options::default();
            options.iterations = 10;
            let res = bench.run(&options, || {
                resize.resize(resize_src_image, &mut dst).unwrap();
            });
            println!("resize - {}: {}", alg_name, res);
        }
    }

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
            let src_image_data = U16::load_big_src_image();
            let src_view = src_image_data.view();
            let mut dst_image = Image::new(new_width, new_height, src_view.pixel_type());
            let mut dst_view = dst_image.view_mut();

            let resize_alg = match alg_name {
                "Nearest" => ResizeAlg::Nearest,
                "Bilinear" => ResizeAlg::Convolution(FilterType::Bilinear),
                "CatmullRom" => ResizeAlg::Convolution(FilterType::CatmullRom),
                "Lanczos3" => ResizeAlg::Convolution(FilterType::Lanczos3),
                _ => return,
            };
            let mut fast_resizer = Resizer::new(resize_alg);

            unsafe {
                fast_resizer.reset_internal_buffers();
                fast_resizer.set_cpu_extensions(cpu_ext);
            }

            #[cfg(not(target_arch = "wasm32"))]
            bench.task(format!("fir {} - {}", ext_name, alg_name), |task| {
                task.iter(|| {
                    fast_resizer.resize(&src_view, &mut dst_view).unwrap();
                })
            });
            #[cfg(target_arch = "wasm32")]
            {
                let mut options = Options::default();
                options.iterations = 10;
                let res = bench.run(&options, || {
                    fast_resizer.resize(&src_view, &mut dst_view).unwrap();
                });
                println!("fir {} - {}: {}", ext_name, alg_name, res);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    utils::print_md_table(bench);
}

#[cfg(not(target_arch = "wasm32"))]
bench_main!("Compare resize of U16 image", bench_downscale_l16,);

#[cfg(target_arch = "wasm32")]
pub fn main() {
    println!("Compare resize of U16 image");
    let mut bench = Bench::new();
    bench_downscale_l16(&mut bench);
}
