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

# ---- pre-reqs ----
python -m pip install --quiet packaging psutil ninja

# ---- install flash-attn ----
TORCH_BEFORE=$(python -c 'import torch; print(torch.__version__)')

echo "Installing flash-attn (this takes 5-10 min) ..."
MAX_JOBS=4 python -m pip install --no-deps flash-attn --no-build-isolation

# ---- guard: torch must not change ----
TORCH_AFTER=$(python -c 'import torch; print(torch.__version__)')
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
    echo "ERROR: torch changed: $TORCH_BEFORE -> $TORCH_AFTER"
    exit 1
fi

python -m pip check

# ---- verify ----
echo ""
echo "=== Verification ==="

python -c "
import torch
from flash_attn import flash_attn_varlen_func
from transformers.utils import is_flash_attn_2_available

assert is_flash_attn_2_available(), 'is_flash_attn_2_available() returned False!'

# smoke test: BF16 varlen kernel with backward
q = torch.randn(100, 8, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
cu = torch.tensor([0, 50, 100], dtype=torch.int32, device='cuda')

out = flash_attn_varlen_func(q, k, v, cu, cu, 50, 50)
assert out.shape == q.shape
assert torch.isfinite(out).all()

out.float().square().mean().backward()
for t in (q, k, v):
    assert t.grad is not None
    assert torch.isfinite(t.grad).all()

print('Flash-Attention varlen smoke test PASSED (fwd + bwd)')
"

echo ""
echo "=== Done ==="
echo "flash-attn installed and verified."
