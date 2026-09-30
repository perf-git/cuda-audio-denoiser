# CUDA Audio Denoiser

This capstone demonstrates a small GPU audio-processing workload: a five-sample moving-average FIR filter suppresses high-frequency interference in a synthetic 440 Hz tone. The program compares a CUDA result against a Python CPU reference and measures signal error before and after filtering.

## GPU implementation

`denoise.py` calls the NVIDIA CUDA Driver API directly through Python's standard-library `ctypes`. It embeds and loads a PTX kernel (`fir5`) that launches one CUDA thread per output sample. Each thread averages five neighboring samples; the first and last two samples are zero-padded. The GPU performs the actual filtering. The CPU implementation exists as a correctness reference and baseline timing.

The included kernel targets NVIDIA GPUs with compute capability 7.5 (Turing) or newer. This machine's run used a GeForce GTX 1650 (compute capability 7.5). The installed NVIDIA driver provides the CUDA Driver API; a separate CUDA Toolkit, `nvcc`, and third-party Python packages are not required. Python 3.9+ is required. On Linux, the NVIDIA driver library must be available as `libcuda.so.1`.

## Run

```powershell
python3 denoise.py
```

Override the input length, GPU launch count, or evidence folder with `--samples`, `--repeats`, and `--out`. The default run processes 32,768 samples at 48 kHz, then writes `evidence/run.log`, `evidence/signal_samples.csv`, and `evidence/waveform.svg`.

The generated CSV includes the clean reference, noisy signal, GPU-filtered output, and CPU-filtered reference for every sample. The log records the GPU name, launch configuration, throughput, RMS error improvement, and maximum GPU-versus-CPU difference. A CPU/GPU comparison tolerance of `2e-6` makes a mismatch fail the run.

## Results

See [`evidence/run.log`](evidence/run.log) for the execution record and [`evidence/waveform.svg`](evidence/waveform.svg) for the output plot. The measured speed includes Python-to-driver launch overhead, so it is an end-to-end demonstration rather than a clean kernel-only benchmark. The short five-tap workload is intentionally small; its main result is correctness and noise suppression, not a claim that GPU processing is faster at every signal size.

## Presentation

The 5–10 minute narration and demonstration outline is in [`PRESENTATION.md`](PRESENTATION.md).
