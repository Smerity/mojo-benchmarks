import benchmark
import sys
from random import rand, seed
from python import Python
import math

alias type = DType.float64
alias bench_size = 2048

"""

def softmax_native(x: list[float]) -> list[float]:
    maxes = max(x)
    x_exp = [math.exp(xi - maxes) for xi in x]
    x_exp_sum = sum(x_exp)
    probs = [xi / x_exp_sum for xi in x_exp]
    return probs
"""


fn softmax[
    len: Int
](x: DTypePointer[type], res_ptr: DTypePointer[type]) -> DTypePointer[type]:
    var max = x[0]
    for i in range(1, len):
        if x[i] > max:
            max = x[i]

    var x_exp = stack_allocation[len, type]()
    var x_exp_sum = 0.0
    for i in range(len):
        x_exp[i] = math.exp(x[i] - max)
        x_exp_sum += x_exp[i]

    for i in range(len):
        res_ptr[i] = x_exp[i] / x_exp_sum

    return res_ptr


fn softmax_simd[len: Int](x: SIMD[type, len]) -> SIMD[type, len]:
    var max = x.reduce_max()

    var x_exp = math.exp(x - max)
    var x_sum = x.reduce_add()

    var probs = x_exp / x_sum

    return probs


def test():
    # import python softmax.py and compare with result of softmax_native(x)
    Python.add_to_path(".")
    Python.add_to_path("benches/softmax")
    var pysoftmax = Python.import_module("softmax")
    var py = Python.import_module("builtins")
    var pyrandom = Python.import_module("random")

    alias testsize = 16

    var mojoin = stack_allocation[testsize, type]()
    var mojosimdin = mojoin.load[width=testsize]()
    var res_mojo = stack_allocation[testsize, type]()

    pyin = py.list()
    for i in range(testsize):
        var val = pyrandom.random()
        pyin.append(val)
        mojoin[i] = val.to_float64()

    var res_py = pysoftmax.softmax_native(pyin)
    var _a = softmax[testsize](mojoin, res_mojo)
    var res_simd_mojo = softmax_simd(mojosimdin)

    for i in range(testsize):
        # acceptable error margin due to float precision
        if math.abs(res_py[i].to_float64() - res_mojo[i]) > 1e-6:
            py.print(py.str("Mismatch at index {}").format(i))
            py.print(
                py.str("Python: {}, Mojo: {}").format(
                    res_py[i].to_float64(), res_mojo[i]
                )
            )
            raise "Test fail"

        # compare mojo with mojo simd now
        if math.abs(res_mojo[i] - res_simd_mojo[i]) > 1e-6:
            py.print(py.str("Mismatch at index {}").format(i))
            py.print(
                py.str("Mojo: {}, Mojo SIMD: {}").format(
                    res_mojo[i], res_simd_mojo[i]
                )
            )
            raise "Test fail"


fn main() raises:
    test()

    var arr = stack_allocation[bench_size, type]()
    # seed(1)
    rand(arr, bench_size)

    var py = Python.import_module("builtins")
    # _ = py.print(py.str("Starting benchmark, size {}...").format(size))

    var dummy = stack_allocation[bench_size, type]()

    var res = softmax[bench_size](arr, dummy)

    @always_inline
    @parameter
    fn worker():
        var bres = softmax[bench_size](arr, dummy)
        benchmark.keep(bres)  # do not optimize out

    var r = benchmark.run[worker](max_runtime_secs=5)
    py.print(py.str("Mean time (naive): {}ms").format(r.mean("ms")))

    # now do simd
    var simd_arr = arr.load[width=bench_size]()
    var res_simd = softmax_simd(simd_arr)

    @always_inline
    @parameter
    fn worker_simd():
        var bres = softmax_simd(simd_arr)
        benchmark.keep(bres)  # do not optimize out

    var r_simd = benchmark.run[worker_simd](max_runtime_secs=5)
    py.print(py.str("Mean time  (SIMD): {}ms").format(r_simd.mean("ms")))
