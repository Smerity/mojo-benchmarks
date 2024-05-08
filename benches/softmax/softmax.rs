#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]

use std::fs::File;
use std::hint::black_box;
use std::io::Read;
use std::mem::MaybeUninit;
use std::simd::num::SimdFloat;
use std::simd::StdFloat;

#[allow(non_upper_case_globals)]
const bench_size: usize = 2048;

use std::simd::f64x8;
type SimdType = f64x8;
const SIMD_SIZE: usize = 8;
const SIMD_COUNT: usize = bench_size / SIMD_SIZE;

use std::arch::x86_64::*;

fn softmax(x: &[f64; bench_size]) -> [f64; bench_size] {
    let mut max = x[0];
    for i in 1..x.len() {
        if x[i] > max {
            max = x[i];
        }
    }

    let mut x_exp = [0.0; bench_size];
    let mut x_exp_sum = 0.0;
    for i in 0..x.len() {
        x_exp[i] = (x[i] - max).exp();
        x_exp_sum += x_exp[i];
    }

    let mut probs = [0.0; bench_size];
    for i in 0..x.len() {
        probs[i] = x_exp[i] / x_exp_sum;
    }

    probs
}

fn softmax_intr(x: &[f64; bench_size]) -> [f64; bench_size] {
    let max = x[0];
    let mut max: __m512d = __m512d::from(SimdType::splat(max));
    for a in x.chunks_exact(8) {
        unsafe {
            let v = _mm512_loadu_pd(a.as_ptr());
            max = _mm512_max_pd(max, v);
        }
    }

    let mut x_exp = [0.0; bench_size];
    let mut x_exp_sum: __m512d = __m512d::from(SimdType::splat(0.0));
    let constant_e: __m512d = __m512d::from(SimdType::splat(std::f64::consts::E));
    for (a, dst) in x.chunks_exact(8).zip(x_exp.chunks_exact_mut(8)) {
        unsafe {
            let v = _mm512_loadu_pd(a.as_ptr());
            let v = _mm512_sub_pd(v, max);

            /// NOTE: What do you mean there's on _mm512_exp_pd??!!??!?!?
            let v = _mm512_mul_pd(constant_e, v);
            //let v = _mm512_pow_pd(constant_e, v);

            _mm512_storeu_pd(dst.as_mut_ptr(), v);
            x_exp_sum = _mm512_add_pd(x_exp_sum, v);
        }
    }
    let x_esp_sum = f64x8::from(x_exp_sum).reduce_sum();
    let x_esp_sum: __m512d = __m512d::from(SimdType::splat(x_esp_sum));

    for dst in x_exp.chunks_exact_mut(8) {
        unsafe {
            let v = _mm512_loadu_pd(dst.as_ptr());
            let v = _mm512_div_pd(v, x_esp_sum);
            _mm512_storeu_pd(dst.as_mut_ptr(), v);
        }
    }

    x_exp
}

fn softmax_simd(x: &[SimdType; SIMD_COUNT]) -> [SimdType; SIMD_COUNT] {
    let max: f64 = *x
        .map(|v| v.reduce_max())
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = SimdType::splat(max);

    let x_exp = x.map(|v| (v - max).exp());
    let x_exp_sum = x_exp.iter().map(|v| v.reduce_sum()).sum();
    let divide_by = SimdType::splat(x_exp_sum);
    x_exp.map(|v| v / divide_by)
}

fn softmax_simd_fold(x: &[SimdType; SIMD_COUNT]) -> [SimdType; SIMD_COUNT] {
    let max = x
        .iter()
        .fold(SimdType::splat(f64::NEG_INFINITY), |acc, v| {
            acc.simd_max(*v)
        })
        .reduce_max();
    let max = SimdType::splat(max);

    let x_exp = x.map(|v| (v - max).exp());
    let x_exp_sum = x_exp.iter().sum::<SimdType>().reduce_sum();
    let divide_by = SimdType::splat(x_exp_sum);

    x_exp.map(|v| v / divide_by)
}

fn softmax_simd_fold_res(x: &[SimdType; SIMD_COUNT], res: &mut [SimdType; SIMD_COUNT]) -> f64 {
    let max = x
        .iter()
        .fold(SimdType::splat(f64::NEG_INFINITY), |acc, v| {
            acc.simd_max(*v)
        })
        .reduce_max();
    let max = SimdType::splat(max);

    let mut x_exp_sum: SimdType = SimdType::splat(0.0);
    for (slot, v) in res.iter_mut().zip(x.iter()) {
        *slot = (*v - max).exp();
        x_exp_sum += *slot;
    }
    let x_exp_sum = x_exp_sum.reduce_sum();

    let divide_by = SimdType::splat(x_exp_sum);

    res.iter_mut().for_each(|v| *v = *v / divide_by);
    // If we don't do this ... it doesn't materialize the array ..?
    // This would be a tad faster maybe - but not by as much as Mojooooo -_-
    res.iter().sum::<SimdType>().reduce_sum()
}

fn main() {
    test();
    let mut random_arr = [0u8; bench_size * 8];
    let mut f = File::open("/dev/urandom").unwrap();
    f.read_exact(&mut random_arr).unwrap();

    let mut arr = [0f64; bench_size];
    // convert random bytes to f64
    for i in 0..bench_size {
        arr[i] = f64::from_le_bytes([
            random_arr[i * 8],
            random_arr[i * 8 + 1],
            random_arr[i * 8 + 2],
            random_arr[i * 8 + 3],
            random_arr[i * 8 + 4],
            random_arr[i * 8 + 5],
            random_arr[i * 8 + 6],
            random_arr[i * 8 + 7],
        ]);
    }

    println!("Length in f64: {}", arr.len());

    softmax(black_box(&arr));

    // println!("starting..");

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    let count = 1000;
    for _ in 0..count {
        black_box(softmax(&arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time (native): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    let count = 1000;
    for _ in 0..count {
        black_box(softmax_intr(&arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time (native): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );

    // prepare array of simd vectors

    let mut simd_arr: [SimdType; SIMD_COUNT] = unsafe {
        #[allow(invalid_value)]
        MaybeUninit::uninit().assume_init()
    };

    for i in 0..SIMD_COUNT {
        simd_arr[i] = SimdType::from_slice(&arr[i * SIMD_SIZE..]);
    }
    println!("Length of SIMD array: {}", simd_arr.len());

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    for _ in 0..count {
        black_box(softmax_simd(&simd_arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time   (SIMD): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    for _ in 0..count {
        black_box(softmax_simd_fold(&simd_arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time   (SIMD fold): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );

    let mut res_arr: [SimdType; SIMD_COUNT] = unsafe {
        #[allow(invalid_value)]
        MaybeUninit::uninit().assume_init()
    };

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    for _ in 0..count {
        black_box(softmax_simd_fold_res(&simd_arr, &mut res_arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time   (SIMD fold res): {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );
}

fn test() {
    // TODO: implement, noop for now
}
