# Coursera submission text

## Project description

CUDA Audio Denoiser applies a five-tap moving-average FIR filter to a synthetic 440 Hz audio signal contaminated with 8 kHz and 12.345 kHz interference. A CUDA PTX kernel, launched through NVIDIA's CUDA Driver API from Python `ctypes`, computes one filtered sample per GPU thread. The output is checked against an independent CPU reference. The program also measures the input and filtered RMS error against the clean signal.

The run on an NVIDIA GeForce GTX 1650 processed 32,768 samples at 48 kHz, improved SNR by 13.90 dB, and differed from the CPU reference by at most 1.2e-7. The exact execution details and generated sample data and waveform plot are included in the project.

## Repository URL

Paste the public repository URL here after publishing the project folder.

## Proof of execution

- `evidence/run.log` records the GPU, launch configuration, timing, RMS error, and CPU comparison.
- `evidence/signal_samples.csv` contains clean, noisy, GPU-filtered, and CPU-reference samples.
- `evidence/waveform.svg` plots the first 100 ms of the noisy input, CUDA output, and clean reference.

## Demonstration

Use `PRESENTATION.md` as the six-minute walkthrough. It demonstrates a fresh run, the CUDA code path, the output plot, the sample CSV, and numerical validation. Record a 5–10 minute screen demonstration using the outline before submitting if the peer-review form expects a recording or presentation link.
