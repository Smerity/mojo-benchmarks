import numpy as np

import time


# Adapted from https://gist.github.com/oysstu/68072c44c02879a2abf94ef350d1c7c6
def crc16(data, poly=0x8408):
    """
    CRC-16-CCITT Algorithm
    """
    crc = 0xFFFF
    for b in data:
        cur_byte = 0xFF & b
        for _ in range(0, 8):
            if (crc & 0x0001) ^ (cur_byte & 0x0001):
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
            cur_byte >>= 1
    crc = ~crc & 0xFFFF
    crc = (crc << 8) | ((crc >> 8) & 0xFF)

    return crc & 0xFFFF


# bench it and get mean time
time_limit = 5
arr_size = 100000


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    data = rng.integers(0, 256, size=(arr_size,), dtype=np.uint8)
    return data


arr = initialize(arr_size)

# warm up
crc16(arr)

# bench
times = []
start = time.time()
while sum(times) < time_limit:
    bstart = time.time()
    crc16(arr)
    now = time.time()
    times.append(now - bstart)
    if now - start > time_limit:
        break

print("Mean time: ", np.mean(times))
