use std::fs::File;
use std::hint::black_box;
use std::io::Read;

fn softmax(x: &[f64]) -> Vec<f64> {
    let mut max = x[0];
    for i in 1..x.len() {
        if x[i] > max {
            max = x[i];
        }
    }

    let mut x_exp = vec![0.0; x.len()];
    let mut x_exp_sum = 0.0;
    for i in 0..x.len() {
        x_exp[i] = (x[i] - max).exp();
        x_exp_sum += x_exp[i];
    }

    let mut probs = vec![0.0; x.len()];
    for i in 0..x.len() {
        probs[i] = x_exp[i] / x_exp_sum;
    }

    probs
}

fn main() {
    test();
    let size = std::env::args().nth(1).unwrap();
    let size = size.parse::<usize>().unwrap();
    // arr is u8 1000000 of random elements allocated on heap
    let mut random_arr = vec![0u8; size * 8];
    let mut f = File::open("/dev/urandom").unwrap();
    f.read_exact(&mut random_arr).unwrap();

    let mut arr = vec![0f64; size];
    // convert random bytes to f64
    for i in 0..size {
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
        "Mean time: {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );
}

fn test() {
    // TODO: implement, noop for now
}
