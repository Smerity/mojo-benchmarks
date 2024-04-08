use std::fs::File;
use std::hint::black_box;
use std::io::Read;

fn crc16(data: &[u8]) -> u16 {
    let poly = 0x8408;
    let mut crc: u16 = 0xffff;

    for byte in data.iter() {
        let mut cur_byte = *byte as u16 & 0xff;
        for _ in 0..8 {
            if (crc & 0x0001) ^ (cur_byte & 0x0001) != 0 {
                crc = (crc >> 1) ^ poly;
            } else {
                crc >>= 1;
            }
            cur_byte >>= 1;
        }
    }

    crc = !crc & 0xffff;
    crc = (crc << 8) | ((crc >> 8) & 0xff);
    crc
}

fn main() {
    test();
    let size = std::env::args().nth(1).unwrap();
    let size = size.parse::<usize>().unwrap();
    // arr is u8 1000000 of random elements allocated on heap
    let mut arr = vec![0u8; size];
    let mut f = File::open("/dev/urandom").unwrap();
    f.read_exact(&mut arr).unwrap();

    crc16(black_box(&arr));

    // println!("starting..");

    // benchmark this 1000 times, get mean
    let start = std::time::Instant::now();
    let count = 1000;
    for _ in 0..count {
        black_box(crc16(&arr));
    }
    let elapsed = start.elapsed().as_nanos();

    println!(
        "Mean time: {}ms",
        elapsed as f64 / 1000.0 / 1000.0 / count as f64
    );
}

fn test() {
    // b"\x31\x32\x33\x34\x35\x36\x37\x38\x39"
    let s = &[0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39];
    assert_eq!(crc16(s), 0x6e90);
}
