import math
import sys
import time
import numpy as np

bench_size = 128


def matmul_np(a, b):
    return np.matmul(a, b)


def matmul_native(a, b):
    m, n = len(a), len(a[0])
    p = len(b[0])

    # init c
    c = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]

    return c


def test():
    x = [[1, 2], [3, 4]]
    y = [[5, 6], [7, 8]]

    x_np = np.array(x)
    y_np = np.array(y)

    res = matmul_native(x, y)
    res_np = matmul_np(x_np, y_np)

    assert np.allclose(np.array(res), res_np)
    assert np.allclose(res, [[19, 22], [43, 50]])


def main():
    test()

    rng = np.random.default_rng(42)
    numpy_in_a = rng.random((bench_size, bench_size))
    numpy_in_b = rng.random((bench_size, bench_size))
    native_in_a = numpy_in_a.tolist()
    native_in_b = numpy_in_b.tolist()

    # warm up
    matmul_native(native_in_a, native_in_b)
    matmul_np(numpy_in_a, numpy_in_b)

    times = 20
    # bench
    start = time.time()
    for i in range(times):
        matmul_native(native_in_a, native_in_b)
    end = time.time()

    print(f"Mean time (native): {((end - start) / times)*1000}ms")

    start = time.time()
    for i in range(times):
        matmul_np(numpy_in_a, numpy_in_b)
    end = time.time()

    print(f"Mean time  (numpy): {((end - start) / times)*1000}ms")


if __name__ == "__main__":
    main()
