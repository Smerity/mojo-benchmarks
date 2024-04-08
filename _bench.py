#!/usr/bin/env python3
# small cmd line util to open benches/{name}/{name}.{suffix}
# and run all of the bench functions in it

import sys
import os


def bench_py(name, size):
    print(f"----- Running {name} - python")
    os.system(f"python3 benches/{name}/{name}.py {size}")


def bench_mojo(name, size):
    print(f"----- Running {name} - mojo")
    os.system(f"mojo benches/{name}/{name}.mojo {size}")


def bench_rust(name, size):
    print(f"----- Running {name} - rust")
    os.system(
        f"rustc benches/{name}/{name}.rs -o benches/{name}/{name}_rs -C opt-level=3 -C target-cpu=native -C lto -C codegen-units=1 -C panic=abort && benches/{name}/{name}_rs {size} && rm benches/{name}/{name}_rs"
    )


size_defaults = {
    "crc16": 100000,
}


def main():
    if len(sys.argv) < 2:
        print("usage: bench.py <name> [size]")
        return

    name = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else size_defaults[name]

    path = os.path.join("benches", name, name + ".py")
    if not os.path.exists(path):
        print("no bench named", name)
        return

    print(f"Benching {name}")
    bench_py(name, size)
    bench_mojo(name, size)
    bench_rust(name, size)

    print(f"----- Done {name}")


if __name__ == "__main__":
    main()
