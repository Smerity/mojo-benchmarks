use std::fs::File;
use std::hint::black_box;
use std::io::Read;

type Data = u8;
#[allow(non_upper_case_globals)]
const bench_size: usize = 1000000;

fn quicksort(data: &mut Vec<Data>, left: isize, right: isize) {
    if left >= right {
        return;
    }

    let pivot = data[right as usize];
    let mut i = left - 1;

    for j in left..right {
        if data[j as usize] <= pivot {
            i += 1;
            data.swap(i as usize, j as usize);
        }
    }

    data.swap((i + 1) as usize, right as usize);

    i += 1;

    quicksort(data, left, i - 1);
    quicksort(data, i + 1, right);
}

fn main() {
    test();
    // arr is u8 1000000 of random elements allocated on heap
    let mut arr: Vec<Data> = vec![0u8; bench_size as usize];
    let mut f = File::open("/dev/urandom").unwrap();
    f.read_exact(arr.as_mut()).unwrap();

    let mut cpy = arr.clone();
    quicksort(&mut cpy, 0, (bench_size - 1) as isize);

    let start = std::time::Instant::now();
    let count = 1000;
    for _ in 0..count {
        let mut cpy = arr.clone();
        quicksort(&mut cpy, 0, (bench_size - 1) as isize);
        let _keep = black_box(cpy);
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time: {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );
}

fn test() {
    let mut arr = vec![4, 3, 2, 1];
    quicksort(&mut arr, 0, 3);
    assert_eq!(arr, vec![1, 2, 3, 4]);
}
