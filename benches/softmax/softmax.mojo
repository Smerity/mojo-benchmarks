import benchmark
import sys
from random import rand, seed
from python import Python
import math

alias type = DType.float64
alias bench_size = 100000

"""

def softmax_native(x: list[float]) -> list[float]:
    maxes = max(x)
    x_exp = [math.exp(xi - maxes) for xi in x]
    x_exp_sum = sum(x_exp)
    probs = [xi / x_exp_sum for xi in x_exp]
    return probs
"""


fn softmax(x: DTypePointer[type], len: Int) -> DTypePointer[type]:
    var max = x[0]
    for i in range(1, len):
        if x[i] > max:
            max = x[i]

    var x_exp = DTypePointer[type].alloc(len)
    var x_exp_sum = 0.0
    for i in range(len):
        x_exp[i] = math.exp(x[i] - max)
        x_exp_sum += x_exp[i]

    var probs = DTypePointer[type].alloc(len)
    for i in range(len):
        probs[i] = x_exp[i] / x_exp_sum

    x_exp.free()
    return probs


def test():
    # import python softmax.py and compare with result of softmax_native(x)
    Python.add_to_path(".")
    Python.add_to_path("benches/softmax")
    var pysoftmax = Python.import_module("softmax")
    var py = Python.import_module("builtins")
    var pyrandom = Python.import_module("random")

    var mojoin = DTypePointer[type].alloc(10)

    pyin = py.list()
    for i in range(10):
        var val = pyrandom.random()
        pyin.append(val)
        mojoin[i] = val.to_float64()

    var res_py = pysoftmax.softmax_native(pyin)
    var res_mojo = softmax(mojoin, 10)

    for i in range(10):
        # acceptable error margin due to float precision
        if math.abs(res_py[i].to_float64() - res_mojo[i]) > 1e-6:
            py.print(py.str("Mismatch at index {}").format(i))
            py.print(
                py.str("Python: {}, Mojo: {}").format(
                    res_py[i].to_float64(), res_mojo[i]
                )
            )
            raise "Test fail"


fn main() raises:
    test()

    var arr = DTypePointer[type].alloc(bench_size)
    # seed(1)
    rand(arr, bench_size)

    var py = Python.import_module("builtins")
    # _ = py.print(py.str("Starting benchmark, size {}...").format(size))

    var res = softmax(arr, bench_size)

    @always_inline
    @parameter
    fn worker():
        var bres = softmax(arr, bench_size)
        benchmark.keep(bres)  # do not optimize out
        bres.free()

    var r = benchmark.run[worker](max_runtime_secs=5)

    arr.free()

    # _ = py.print(py.str("Result: {}, iters: {}").format(res, iters))

    py.print(py.str("Mean time: {}ms").format(r.mean("ms")))
