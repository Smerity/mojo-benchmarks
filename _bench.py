#!/usr/bin/env python3
# small cmd line util to open benches/{name}/{name}.{suffix}
# and run all of the bench functions in it

import sys
import os
from typing import Literal


def compile_size(
    name,
    size,
    format: Literal["mojo", "py", "rs"],
):
    to_match = "bench_size = "
    if format == "rs":
        to_match = "bench_size: usize = "

    # open file under /tmp, find line that has `size = ` in it and replace the value with the new size
    with open(f"benches/{name}/{name}.{format}", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if to_match in line and not line[0].isspace():
                words = line.split()
                do_semicolon = format == "rs"
                words[-1] = f"{size}{';' if do_semicolon else ''}"
                lines[i] = " ".join(words) + "\n"

    with open(f"tmp/{name}.{format}", "w") as f:
        f.writelines(lines)


def bench_py(name, size):
    print(f"----- Running {name} - python")
    # copy file
    os.system(f"cp benches/{name}/{name}.py tmp/")
    compile_size(name, size, "py")
    os.system(f"python3 tmp/{name}.py {size}")


def bench_mojo(name, size):
    print(f"----- Running {name} - mojo")
    os.system(f"cp benches/{name}/{name}.mojo tmp/")
    compile_size(name, size, "mojo")
    os.system(f"mojo tmp/{name}.mojo {size}")


def bench_rust(name, size):
    print(f"----- Running {name} - rust")
    os.system(f"cp benches/{name}/{name}.rs tmp/")
    compile_size(name, size, "rs")
    os.system(
        f"rustc tmp/{name}.rs -o tmp/{name}_rs -C opt-level=3 -C target-cpu=native -C lto -C codegen-units=1 -C panic=abort && tmp/{name}_rs {size} && rm tmp/{name}_rs"
    )


size_defaults = {
    "crc16": 100000,
    "quicksort": 10000,
    "softmax": 10000,
}


def do_bench(name, size):
    path = os.path.join("benches", name, name + ".py")
    if not os.path.exists(path):
        print("no bench named", name)
        return

    print(f"Benching {name} (size {size})")
    bench_py(name, size)
    bench_mojo(name, size)
    bench_rust(name, size)

    print(f"----- Done {name}")


def main():
    if len(sys.argv) < 2:
        # bench all
        print("Benching all")

        for name in os.listdir("benches"):
            if name.startswith("."):
                continue

            do_bench(name, size_defaults[name])
            print("\n--------------------\n")

        print("Done all!")
        return

    name = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else size_defaults[name]

    do_bench(name, size)


if __name__ == "__main__":
    main()
