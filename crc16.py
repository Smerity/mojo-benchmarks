from math import e
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


arr_size = 100000


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    data = rng.integers(0, 256, size=(arr_size,), dtype=np.uint8)
    return data


arr = initialize(arr_size)

# warm up
crc16(arr)

print("CRC16: ", crc16(arr))
times = 100
# bench
start = time.time()
for i in range(times):
    crc16(arr)

end = time.time()
print("Mean time: ", (end - start) / times)
