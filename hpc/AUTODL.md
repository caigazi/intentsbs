# AutoDL handoff

Use the development branch `codex/gpu-teacher-dev`. The current purpose of
the cloud instance is backend verification, not formal dataset generation.

```bash
git clone -b codex/gpu-teacher-dev https://github.com/caigazi/sbs.git
cd sbs
bash hpc/autodl_setup.sh
bash hpc/autodl_smoke.sh
```

For direct interactive JAX commands, first run:

```bash
source hpc/autodl_env.sh
```

Expected setup output contains `backend: gpu`, one `GpuDevice`, and `x64:
True`. It must also resolve `ptxas` inside the active Python environment at
`site-packages/nvidia/cuda_nvcc/bin/ptxas`, currently pinned to CUDA 12.9.86.
The smoke runs all unit tests, a batched Wang-QP parity benchmark, and only the
N=2 Teacher case.

Do not run N=8 or generate training data yet. Under the current one-QP-per-
30-ms sampled-data backend, the N=8 two-stream case leaves the Wang braking
certificate while remaining just outside the public 0.20-m center-distance
line. The safety update architecture must be frozen first.
