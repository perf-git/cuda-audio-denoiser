# Capstone demonstration (about 6 minutes)

## 0:00–0:45 — Problem

“This project shows how a GPU can help with an audio-processing task. The input is a 440 Hz tone contaminated by two higher-frequency signals. I use a five-sample moving-average FIR filter to smooth out some of that interference.”

## 0:45–2:00 — GPU path

Open `denoise.py` and point out that it uses Python's `ctypes` to call the NVIDIA CUDA Driver API directly. The program loads the embedded PTX kernel, copies samples to device memory, launches 256-thread blocks, synchronizes, and copies the filtered samples back. Each thread computes one output sample from five neighboring inputs. The CPU filter is included as a reference for correctness.

## 2:00–3:00 — Run

Run `python3 denoise.py` in a terminal. Show the GPU name in `evidence/run.log`, the success result, and the generated artifacts. Explain that the run uses the installed NVIDIA driver and needs no CUDA Toolkit or Python GPU package.

## 3:00–4:15 — Quality

Open `evidence/waveform.svg`. Compare the noisy input with the CUDA output and clean reference. Then show the log's RMS error and SNR improvement. The program also compares every GPU sample against the CPU reference; the maximum difference must stay below `2e-6` or the program exits as a failure.

## 4:15–5:15 — Results and limits

Show `evidence/signal_samples.csv`, which preserves all input and output samples. Explain that the throughput timing includes host-to-driver launch overhead and that a five-tap workload is small. This demonstrates the CUDA execution path and numerical agreement; it does not imply a GPU speedup for every signal size. Larger batches, many channels, or longer filters would make a more representative performance study.

## 5:15–6:00 — Close

“The deliverable includes the source, exact run command, execution log, full sample CSV, waveform plot, and this demonstration outline. The core idea is to make GPU use explicit and reproducible while checking its output against an independent CPU calculation.”
