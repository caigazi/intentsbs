# AutoDL handoff

Use the clean GitHub `main` branch. The cloud instance is the compute worker for
Gate-0 verification and Teacher-distillability experiments, not formal dataset
generation yet.

```bash
git clone -b main https://github.com/caigazi/intentsbs.git IntentComm-SBS-Clean
cd IntentComm-SBS-Clean
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
The clean AutoDL clone has passed all 31 tests. The V100 JAX backend runs with
x64 enabled and matches the NumPy Wang-QP reference to numerical precision.

After the hybrid Eq. (17) repair, N=2/N=5/N=8 targeted safety smoke cases all
pass with zero certificate violations. Do not generate the 48-scene dataset
yet: finish protocol freeze, then pass Teacher distillability and the tiny
closed-loop overfit Gate first.
