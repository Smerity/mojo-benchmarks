import benchmark
from random import rand
from python import Python

alias type = DType.uint8
alias size = 100000


fn crc16_naive[len: Int, poly: Int](data: DTypePointer[type]) -> Int:
    # CRC-16-CCITT Algorithm

    var crc = 0xFFFF

    for b in range(len):
        var cur_byte = 0xFF & data[b]

        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1

    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


fn main() raises:
    let arr = DTypePointer[type].alloc(size)
    rand(arr, size)

    let py = Python.import_module("builtins")
    _ = py.print(py.str("Starting benchmark, size {}...").format(size))

    var res = 0

    @always_inline
    @parameter
    fn worker():
        res = crc16_naive[size, 0xFFFF](arr)

    let r = benchmark.run[worker](max_runtime_secs=5)

    _ = py.print(py.str("Result: {}").format(res))

    r.print()
