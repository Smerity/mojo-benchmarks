import sys
import time
from random import rand, seed
from python import Python
import memory
import benchmark

alias type = DType.int8


fn quicksort(inout data: DTypePointer[type], left: Int, right: Int):
    if left >= right:
        return

    var pivot = data[right]
    var i = left - 1

    for j in range(left, right):
        if data[j] <= pivot:
            i = i + 1
            var tmp = data[i]
            data[i] = data[j]
            data[j] = tmp

    var tmp = data[i + 1]
    data[i + 1] = data[right]
    data[right] = tmp

    i += 1

    quicksort(data, left, i - 1)
    quicksort(data, i + 1, right)


def test():
    var testlen = 100
    var data = DTypePointer[type].alloc(testlen)
    rand(data, testlen)

    quicksort(data, 0, testlen - 1)

    for i in range(testlen - 1):
        if data[i] > data[i + 1]:
            raise "Sort failed"


fn main() raises:
    test()
    # size is 1 arg from sys
    var size = StringRef.__int__(sys.argv()[1])
    var arr = DTypePointer[type].alloc(size)
    rand(arr, size)

    var py = Python.import_module("builtins")

    @always_inline
    @parameter
    fn worker():
        var temp = DTypePointer[type].alloc(size)
        memcpy(temp, arr, size)
        quicksort(temp, 0, size - 1)
        temp.free()

    var r = benchmark.run[worker](max_runtime_secs=5)

    py.print(py.str("Mean time: {}ms").format(r.mean("ms")))
