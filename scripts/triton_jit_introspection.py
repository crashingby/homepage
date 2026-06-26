#!/usr/bin/env python3
"""
Observe what @triton.jit returns.

This script is intentionally written as a learning note helper:

    /home/huangxy/miniconda3/envs/triton/bin/python scripts/triton_jit_introspection.py

It does not launch the GPU kernel by default.  The goal is to inspect the
Python-side object returned by @triton.jit and connect it back to
triton/runtime/jit.py.
"""

from __future__ import annotations

import inspect
from typing import Any

import triton
import triton.language as tl
from triton.runtime.jit import JITCallable, JITFunction, KernelInterface


def custom_repr(_kernel: Any) -> str:
    # This function is passed to @triton.jit(repr=...).
    # Later, kernel.repr(None) should return this string.
    return "custom_repr_vector_add"


@triton.jit
def add_kernel_plain(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a normal Triton kernel.  The important point here is not the
    # vector add itself, but the fact that @triton.jit replaces this Python
    # function with a JITFunction object at module import time.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit(
    repr=custom_repr,
    do_not_specialize=["n_elements"],
    do_not_specialize_on_alignment=["x_ptr", "y_ptr", "out_ptr"],
    debug=False,
    noinline=False,
)
def add_kernel_with_options(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel is intentionally identical to add_kernel_plain.
    # The difference is only in @triton.jit(...) options, so we can observe
    # where those options are stored on the returned JITFunction object.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def raw_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This function is not decorated here.  We manually pass it to triton.jit
    # below to compare triton.jit(fn) and triton.jit(...)(fn).
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def show(label: str, value: Any) -> None:
    print(f"{label:<42} {value!r}")


def safe_show(label: str, getter) -> None:
    try:
        show(label, getter())
    except Exception as exc:
        print(f"{label:<42} {type(exc).__name__}: {exc}")


def inspect_jit_function(name: str, kernel: JITFunction) -> None:
    print()
    print("=" * 88)
    print(name)
    print("=" * 88)

    # Class relationship: this corresponds to JITFunction(JITCallable,
    # KernelInterface[T]) in triton/runtime/jit.py.
    show("object", kernel)
    show("type(object)", type(kernel))
    show("isinstance(JITFunction)", isinstance(kernel, JITFunction))
    show("isinstance(JITCallable)", isinstance(kernel, JITCallable))
    show("isinstance(KernelInterface)", isinstance(kernel, KernelInterface))

    print()
    print("-- wrapped Python function metadata --")
    # These fields come from JITCallable.__init__.
    safe_show(".fn", lambda: kernel.fn)
    safe_show(".signature", lambda: kernel.signature)
    safe_show(".src[:100]", lambda: kernel.src[:100].replace("\n", "\\n"))
    safe_show(".raw_src[0]", lambda: kernel.raw_src[0].rstrip())
    safe_show(".starting_line_number", lambda: kernel.starting_line_number)

    print()
    print("-- JITFunction configuration fields --")
    # These fields come from JITFunction.__init__.
    for attr in [
        "do_not_specialize",
        "do_not_specialize_on_alignment",
        "_repr",
        "debug",
        "noinline",
        "arg_names",
        "constexprs",
        "pre_run_hooks",
    ]:
        safe_show(f".{attr}", lambda attr=attr: getattr(kernel, attr))

    print()
    print("-- KernelParam entries --")
    # Each function parameter becomes a KernelParam.  This is where
    # tl.constexpr, do_not_specialize, and do_not_specialize_on_alignment
    # become per-parameter metadata.
    for param in kernel.params:
        print(f"param[{param.num}] {param.name!r}")
        for attr in [
            "annotation",
            "is_constexpr",
            "do_not_specialize",
            "do_not_specialize_on_alignment",
        ]:
            safe_show(f"  .{attr}", lambda attr=attr, param=param: getattr(param, attr))

    print()
    print("-- methods and launch syntax --")
    safe_show("kernel.repr(None)", lambda: kernel.repr(None))
    safe_show("type(kernel.parse())", lambda: type(kernel.parse()))
    safe_show("kernel.cache_key[:32]", lambda: kernel.cache_key[:32])

    # Direct call goes to JITFunction.__call__, which intentionally raises.
    safe_show("direct kernel(...) call", lambda: kernel(None, None, None, 0, BLOCK_SIZE=128))

    # Bracket syntax goes to KernelInterface.__getitem__ and returns a launcher
    # closure that remembers grid.
    grid = lambda meta: (triton.cdiv(1024, meta["BLOCK_SIZE"]),)
    launcher = kernel[grid]
    show("kernel[grid]", launcher)
    show("type(kernel[grid])", type(launcher))
    show("callable(kernel[grid])", callable(launcher))


def main() -> None:
    print("triton version:", triton.__version__)

    print()
    print("=" * 88)
    print("Decorator entry points")
    print("=" * 88)

    direct = triton.jit(raw_kernel)
    decorator = triton.jit(do_not_specialize=["n_elements"])
    via_decorator = decorator(raw_kernel)

    show("@triton.jit result type", type(add_kernel_plain))
    show("triton.jit(fn) result type", type(direct))
    show("triton.jit(...)", decorator)
    show("type(triton.jit(...))", type(decorator))
    show("triton.jit(...)(fn) result type", type(via_decorator))
    show("same raw function wrapped twice?", direct.fn is via_decorator.fn)
    show("different wrapper objects?", direct is not via_decorator)

    inspect_jit_function("@triton.jit", add_kernel_plain)
    inspect_jit_function("@triton.jit(...options...)", add_kernel_with_options)

    print()
    print("Run a real launch separately after understanding the object model.")
    print("For example, extend this script with torch CUDA tensors and call kernel[grid](...).")


if __name__ == "__main__":
    main()
