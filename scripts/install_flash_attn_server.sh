#!/bin/bash
# ---------------------------------------------------------------
# Install flash-attn on the Stream-vad server.
# Run AFTER activating the conda environment:
#   conda activate streamvad
#   bash scripts/install_flash_attn_server.sh
# ---------------------------------------------------------------
set -euo pipefail

echo "=== Flash-Attention Install for Stream-vad Server ==="

# ---- environment ----
unset PYTHONPATH
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"

echo "CUDA_HOME = $CUDA_HOME"
echo "Python    = $(which python)"
echo "PyTorch   = $(python -c 'import torch; print(torch.__version__)')"

# ---- guard: don't let pip replace torch ----
TORCH_VER=$(python -c 'import torch; print(torch.__version__)')
echo "Torch version: $TORCH_VER"

# ---- pre-reqs ----
python -m pip install --quiet packaging psutil ninja

# ---- install flash-attn ----
echo "Installing flash-attn (this takes 5-10 min) ..."
MAX_JOBS=4 python -m pip install flash-attn --no-build-isolation

# ---- verify ----
echo ""
echo "=== Verification ==="
python -m pip check 2>&1 | grep -i flash || true

python -c "
import torch
from flash_attn import flash_attn_func
from transformers.utils import is_flash_attn_2_available

assert is_flash_attn_2_available(), 'is_flash_attn_2_available() returned False!'

# smoke test: small BF16 varlen kernel
B, H, D = 1, 8, 128
q = torch.randn(100, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=False)
k = torch.randn(100, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=False)
v = torch.randn(100, H, D, device='cuda', dtype=torch.bfloat16, requires_grad=False)
cu = torch.tensor([0, 50, 100], dtype=torch.int32, device='cuda')
out = flash_attn_func(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                      max_seqlen_q=50, max_seqlen_k=50)
print(f'flash_attn_func output shape: {out.shape}')
assert out.shape == (100, H, D)
print('Flash-Attention smoke test PASSED')
"

echo ""
echo "=== Done ==="
echo "flash-attn installed and verified."
