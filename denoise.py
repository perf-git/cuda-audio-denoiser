#!/usr/bin/env python3
"""A dependency-free CUDA Driver API demo: 5-tap GPU audio denoising."""
from __future__ import annotations

import argparse
import array
import ctypes
import csv
import math
import os
import sys
import time
from pathlib import Path

N = 32_768
SAMPLE_RATE = 48_000
REPEATS = 40

PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry fir5(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 count
)
{
    .reg .pred %p<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<9>;
    .reg .f32 %f<6>;
    ld.param.u64 %rd1, [input_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u32 %r2, [count];
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.u32 %r1, %r3, %r4, %r5;
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra DONE;
    mul.wide.u32 %rd3, %r1, 4;
    add.u64 %rd4, %rd2, %rd3;
    setp.lt.u32 %p2, %r1, 2;
    @%p2 bra ZERO;
    sub.u32 %r3, %r2, 2;
    setp.ge.u32 %p2, %r1, %r3;
    @%p2 bra ZERO;
    add.u64 %rd5, %rd1, %rd3;
    sub.u64 %rd6, %rd5, 8;
    ld.global.f32 %f1, [%rd6];
    sub.u64 %rd6, %rd5, 4;
    ld.global.f32 %f2, [%rd6];
    ld.global.f32 %f3, [%rd5];
    add.u64 %rd6, %rd5, 4;
    ld.global.f32 %f4, [%rd6];
    add.u64 %rd6, %rd5, 8;
    ld.global.f32 %f5, [%rd6];
    add.f32 %f1, %f1, %f2;
    add.f32 %f1, %f1, %f3;
    add.f32 %f1, %f1, %f4;
    add.f32 %f1, %f1, %f5;
    mul.f32 %f1, %f1, 0f3E4CCCCD;
    st.global.f32 [%rd4], %f1;
    bra DONE;
ZERO:
    mov.f32 %f1, 0f00000000;
    st.global.f32 [%rd4], %f1;
DONE:
    ret;
}
"""


class Cuda:
    """Small ctypes binding to the CUDA Driver API; no Python GPU package needed."""

    def __init__(self) -> None:
        try:
            self.lib = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        except OSError as exc:
            raise RuntimeError(f"CUDA driver library unavailable: {exc}") from exc

        self.call("cuInit", 0)
        device = ctypes.c_int()
        self.call("cuDeviceGet", ctypes.byref(device), 0)
        name = ctypes.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), device)
        self.name = name.value.decode("utf-8", errors="replace")
        self.context = ctypes.c_void_p()
        self.call("cuCtxCreate_v2", ctypes.byref(self.context), 0, device)

        module = ctypes.c_void_p()
        ptx_buffer = ctypes.create_string_buffer(PTX.encode("ascii"))
        self.call("cuModuleLoadData", ctypes.byref(module), ctypes.cast(ptx_buffer, ctypes.c_void_p))
        self.module = module
        function = ctypes.c_void_p()
        self.call("cuModuleGetFunction", ctypes.byref(function), module, b"fir5")
        self.function = function

    def _fn(self, name: str):
        function = getattr(self.lib, name)
        function.restype = ctypes.c_int
        u64p = ctypes.POINTER(ctypes.c_uint64)
        vp = ctypes.POINTER(ctypes.c_void_p)
        signatures = {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
            "cuDeviceGetName": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
            "cuCtxCreate_v2": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
            "cuCtxDestroy_v2": [ctypes.c_void_p],
            "cuModuleLoadData": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
            "cuModuleGetFunction": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p],
            "cuModuleUnload": [ctypes.c_void_p],
            "cuMemAlloc_v2": [u64p, ctypes.c_size_t],
            "cuMemFree_v2": [ctypes.c_uint64],
            "cuMemcpyHtoD_v2": [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
            "cuMemcpyDtoH_v2": [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
            "cuCtxSynchronize": [],
            "cuLaunchKernel": [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, vp, vp],
        }
        function.argtypes = signatures.get(name)
        return function

    def call(self, name: str, *args) -> None:
        result = self._fn(name)(*args)
        if result != 0:
            error_name = ctypes.c_char_p()
            error_fn = getattr(self.lib, "cuGetErrorName")
            error_fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            error_fn.restype = ctypes.c_int
            error_fn(result, ctypes.byref(error_name))
            label = error_name.value.decode("ascii", errors="replace") if error_name.value else "unknown"
            raise RuntimeError(f"CUDA Driver API {name} failed: {label} ({result})")

    def filter(self, source: array.array, repeats: int) -> tuple[array.array, float]:
        n = len(source)
        byte_count = source.itemsize * n
        device_input = ctypes.c_uint64()
        device_output = ctypes.c_uint64()
        self.call("cuMemAlloc_v2", ctypes.byref(device_input), byte_count)
        self.call("cuMemAlloc_v2", ctypes.byref(device_output), byte_count)
        try:
            source_ptr = ctypes.c_void_p(source.buffer_info()[0])
            self.call("cuMemcpyHtoD_v2", device_input, source_ptr, byte_count)
            count = ctypes.c_uint32(n)
            params = (ctypes.c_void_p * 3)(
                ctypes.cast(ctypes.byref(device_input), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(device_output), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(count), ctypes.c_void_p),
            )
            blocks = (n + 255) // 256
            start = time.perf_counter()
            for _ in range(repeats):
                self.call("cuLaunchKernel", self.function, blocks, 1, 1, 256, 1, 1, 0,
                          ctypes.c_void_p(), params, ctypes.c_void_p())
            self.call("cuCtxSynchronize")
            elapsed = time.perf_counter() - start
            result = array.array("f", [0.0]) * n
            self.call("cuMemcpyDtoH_v2", ctypes.c_void_p(result.buffer_info()[0]), device_output, byte_count)
            return result, elapsed
        finally:
            self.call("cuMemFree_v2", device_output)
            self.call("cuMemFree_v2", device_input)

    def close(self) -> None:
        self.call("cuModuleUnload", self.module)
        self.call("cuCtxDestroy_v2", self.context)


def make_signal(n: int) -> tuple[array.array, array.array]:
    clean, noisy = array.array("f"), array.array("f")
    for i in range(n):
        t = i / SAMPLE_RATE
        base = 0.7 * math.sin(2 * math.pi * 440 * t)
        interference = 0.25 * math.sin(2 * math.pi * 8_000 * t) + 0.08 * math.sin(2 * math.pi * 12_345 * t)
        clean.append(base)
        noisy.append(base + interference)
    return clean, noisy


def cpu_filter(source: array.array) -> array.array:
    result = array.array("f", [0.0]) * len(source)
    for i in range(2, len(source) - 2):
        result[i] = (source[i - 2] + source[i - 1] + source[i] + source[i + 1] + source[i + 2]) * 0.2
    return result


def rms_error(reference: array.array, signal: array.array, start: int = 2, end: int | None = None) -> float:
    end = end if end is not None else len(reference) - 2
    return math.sqrt(sum((reference[i] - signal[i]) ** 2 for i in range(start, end)) / (end - start))


def write_evidence(folder: Path, clean: array.array, noisy: array.array, gpu: array.array, cpu: array.array,
                   gpu_s: float, cpu_s: float, gpu_name: str, max_error: float, repeats: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "signal_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("sample", "time_seconds", "clean_reference", "noisy_input", "gpu_filtered", "cpu_reference"))
        for i, (c, x, y, z) in enumerate(zip(clean, noisy, gpu, cpu)):
            writer.writerow((i, f"{i / SAMPLE_RATE:.8f}", f"{c:.8f}", f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"))

    before = rms_error(clean, noisy)
    after = rms_error(clean, gpu)
    improvement = 20 * math.log10(before / after)
    report = (
        "CUDA 5-tap audio denoiser execution report\n"
        f"GPU: {gpu_name}\nCUDA path: NVIDIA CUDA Driver API; embedded PTX kernel fir5; 256 threads/block\n"
        f"Samples: {len(noisy)} at {SAMPLE_RATE} Hz; launches timed: {repeats}\n"
        f"GPU launch batch: {gpu_s:.6f} s ({len(noisy) * repeats / gpu_s:,.0f} sample outputs/s including launch overhead)\n"
        f"CPU reference: {cpu_s:.6f} s ({len(noisy) / cpu_s:,.0f} samples/s)\n"
        f"Input noise RMS: {before:.6f}; filtered RMS error: {after:.6f}; SNR improvement: {improvement:.2f} dB\n"
        f"GPU vs CPU max absolute difference: {max_error:.9g}\n"
    )
    (folder / "run.log").write_text(report, encoding="utf-8")

    # A lightweight vector plot of a decimated 100 ms slice; no plotting package required.
    width, height, pad = 1000, 460, 34
    first, last = 0, min(len(noisy), SAMPLE_RATE // 10)
    indices = list(range(first, last, max(1, (last - first) // 1000)))
    lo, hi = -1.15, 1.15

    def points(values: array.array) -> str:
        coords = []
        for j in indices:
            x = pad + j / max(1, last - 1) * (width - 2 * pad)
            y = pad + (hi - values[j]) / (hi - lo) * (height - 2 * pad)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#101827"/><text x="34" y="26" fill="#f1f5f9" font-family="sans-serif" font-size="18">CUDA 5-tap FIR audio denoiser · first 100 ms</text>
<polyline points="{points(noisy)}" fill="none" stroke="#fb7185" stroke-width="1.1" opacity="0.8"/>
<polyline points="{points(gpu)}" fill="none" stroke="#38bdf8" stroke-width="1.8"/>
<polyline points="{points(clean)}" fill="none" stroke="#a3e635" stroke-width="1.1" opacity="0.9"/>
<text x="700" y="28" fill="#fb7185" font-family="sans-serif" font-size="13">Noisy input</text><text x="800" y="28" fill="#38bdf8" font-family="sans-serif" font-size="13">CUDA output</text><text x="900" y="28" fill="#a3e635" font-family="sans-serif" font-size="13">Clean signal</text>
<text x="34" y="445" fill="#cbd5e1" font-family="sans-serif" font-size="13">Synthetic 440 Hz tone + 8 kHz and 12.345 kHz interference · 48 kHz sample rate</text></svg>'''
    (folder / "waveform.svg").write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=N)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    args = parser.parse_args()
    if args.samples < 5 or args.repeats < 1:
        parser.error("--samples must be >= 5 and --repeats must be >= 1")

    clean, noisy = make_signal(args.samples)
    cpu_start = time.perf_counter()
    cpu = cpu_filter(noisy)
    cpu_s = time.perf_counter() - cpu_start

    cuda = None
    try:
        cuda = Cuda()
        gpu, gpu_s = cuda.filter(noisy, args.repeats)
        max_error = max(abs(a - b) for a, b in zip(gpu, cpu))
        if max_error > 2e-6:
            raise RuntimeError(f"GPU result failed CPU comparison: max difference {max_error}")
        write_evidence(args.out, clean, noisy, gpu, cpu, gpu_s, cpu_s, cuda.name, max_error, args.repeats)
        print((args.out / "run.log").read_text(encoding="utf-8"), end="")
        print(f"Evidence written to: {args.out.resolve()}")
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if cuda is not None:
            try:
                cuda.close()
            except RuntimeError as exc:
                print(f"CUDA cleanup note: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
