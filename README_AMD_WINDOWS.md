# SpargeAttn - AMD ROCm on Windows Setup Guide

This guide explains how to build and run SpargeAttn on Windows with AMD GPUs using ROCm.

> **Note:** These steps should also work on Linux with minor modifications (use bash commands instead of PowerShell, `source venv/bin/activate` instead of `.\venv\Scripts\Activate.ps1`, and skip the Visual Studio environment setup). However, Linux support has not been tested yet and may have issues.

## Supported Hardware

SpargeAttn on Windows has been tested with RDNA3/RDNA3.5 GPUs (gfx1100, gfx1101, gfx1102, gfx1103, gfx1151).

## Prerequisites

- Windows 10/11
- Python 3.11, 3.12, or 3.13
- Visual Studio 2022 with C++ build tools
- AMD Adrenaline driver (latest recommended)

## Installation

### 1. Install ROCm and PyTorch from TheRock

Follow the instructions at [ROCm/TheRock RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) to install ROCm and PyTorch wheels for your GPU architecture.

#### Create a Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Install PyTorch (includes ROCm SDK as dependency)

For **gfx1151** (AMD Strix Halo iGPU):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ --pre torch torchaudio torchvision
```

For **gfx110X** (RX 7900 XTX, RX 7800 XT, RX 7700S, Radeon 780M):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ --pre torch torchaudio torchvision
```

For **gfx120X** (RX 9060, RX 9070):
```powershell
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ --pre torch torchaudio torchvision
```

#### Initialize ROCm SDK

```powershell
rocm-sdk init
```

#### Install Triton with AMD Windows Support

```powershell
pip install triton-windows
```

### 2. Set Environment Variables

Open a PowerShell terminal and run:

```powershell
# Activate Visual Studio environment
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Set ROCm paths using rocm-sdk
$ROCM_ROOT = (rocm-sdk path --root).Trim()
$ROCM_BIN = (rocm-sdk path --bin).Trim()
$env:ROCM_HOME = $ROCM_ROOT
$env:PATH = "$ROCM_ROOT\lib\llvm\bin;$ROCM_BIN;$env:PATH"

# Set compiler and build settings
$env:CC = "clang-cl"
$env:CXX = "clang-cl"
$env:DISTUTILS_USE_SDK = "1"

# Enable experimental features
$env:FLASH_ATTENTION_TRITON_AMD_ENABLE = "TRUE"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
```

### 3. Build and Install SpargeAttn

```powershell
cd <path_to_spargeattn>
pip install --no-build-isolation -v .
```

## Testing

### Quick Correctness Test

Run this script to verify SpargeAttn is working correctly by comparing against PyTorch SDPA:

```python
import torch
import torch.nn.functional as F
from spas_sage_attn.core import spas_sage_attn_meansim_cuda

device = torch.device('cuda')

# Create random test tensors (use float16 for ROCm compatibility)
q = torch.randn(1, 12, 2048, 128, dtype=torch.float16, device=device)
k = torch.randn(1, 12, 2048, 128, dtype=torch.float16, device=device)
v = torch.randn(1, 12, 2048, 128, dtype=torch.float16, device=device)

# Compute reference output using PyTorch SDPA
with torch.no_grad():
    sdpa = F.scaled_dot_product_attention(q.float(), k.float(), v.float()).to(torch.float16)

# Compute SpargeAttn output (with 100% sparsity = dense attention)
sparge = spas_sage_attn_meansim_cuda(
    q, k, v,
    is_causal=False,
    smooth_k=False,
    simthreshd1=0.0,   # No similarity threshold (keep all blocks)
    cdfthreshd=1.0,    # 100% sparsity
    pvthreshd=0,
    tensor_layout='HND'
)

# Compare outputs using cosine similarity
cos = F.cosine_similarity(
    sdpa.flatten().float().unsqueeze(0),
    sparge.flatten().float().unsqueeze(0)
)
print(f'Cosine similarity: {cos.item():.6f}')  # Should be ~0.9999
```

Save this as `test_spargeattn.py` and run:

```powershell
python test_spargeattn.py
```

Expected output:
```
Cosine similarity: 0.999900
```

A cosine similarity above 0.999 indicates the kernel is working correctly.

## Performance Notes

At L=4096, D=128, bf16 vs PyTorch SDPA (with aotriton):

| Sparsity | Time | Speedup vs SDPA |
|----------|------|-----------------|
| 100% | 33.0 ms | 0.18x |
| 50% | 13.7 ms | 0.43x |
| 25% | 7.4 ms | 0.79x |
| **10%** | **3.2 ms** | **1.81x** |
| 5% | 1.8 ms | 3.26x |
| 2% | 1.0 ms | 6.07x |

**Break-even point**: ~20-25% sparsity. Below that, SpargeAttn is faster than dense SDPA.

## Known Issues

1. **No FP8 support on RDNA3** - rocWMMA on gfx11xx doesn't support FP8, so FP16/BF16 is used for V.

2. **Triton compiler warnings** - You may see `clang-cl: warning: unknown argument ignored` warnings during first run. These are harmless.

## Troubleshooting

### "LoadLibrary failed" or "cannot find amdhip64.dll"

Make sure you ran `rocm-sdk init` after installing the ROCm SDK packages.

### "LINK : fatal error LNK1104: cannot open file 'python312.lib'"

Ensure Visual Studio environment is activated before building:
```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && set' | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') } }
```

### "PermissionError" when compiling Triton kernels

This is a known Windows issue with temp file handling. Make sure you're using the latest `triton-windows` package (`pip install --upgrade triton-windows`).

