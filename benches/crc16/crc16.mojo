import benchmark
import sys
from random import rand, seed
from python import Python

alias type = DType.int8


fn print(s: String):
    try:
        var py = Python.import_module("builtins")
        _ = py.print(py.str(s))
    except e:
        pass


@always_inline
fn crc16_naive[poly: Int](data: DTypePointer[type], len: Int) -> Int:
    # CRC-16-CCITT Algorithm
    # naively ported from python version

    var crc = 0xFFFF

    for b in range(len):
        var cur_byte = 0xFF & data[b]

        @unroll
        for _ in range(8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1

    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


fn test() raises:
    # s = b"\x31\x32\x33\x34\x35\x36\x37\x38\x39"
    var s: StringLiteral = "123456789"
    var data: DTypePointer[type] = s.data()

    # print as asci
    var crc = crc16_naive[0x8408](data, len(s))

    if not crc == 0x6E90:
        raise "Test failed"


fn main() raises:
    test()
    # size is 1 arg from sys
    var size = StringRef.__int__(sys.argv()[1])
    var arr = DTypePointer[type].alloc(size)
    # seed(1)
    rand(arr, size)

    var py = Python.import_module("builtins")
    # _ = py.print(py.str("Starting benchmark, size {}...").format(size))

    var res = 0
    res = crc16_naive[0xFFFF](arr, size)
    var tainted = False
    var iters = 0

    @always_inline
    @parameter
    fn worker():
        var bres = crc16_naive[0xFFFF](arr, size)
        benchmark.keep(bres)  # do not optimize out

    var r = benchmark.run[worker](max_runtime_secs=5)

    if tainted:
        raise "Tainted result"

    # _ = py.print(py.str("Result: {}, iters: {}").format(res, iters))

    py.print(py.str("Mean time: {}ms").format(r.mean("ms")))
