"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
from pathlib import Path
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY_DIR = os.path.join(THIS_DIR, "third_party")

HAS_SM90 = False
SAGE2PP_ENABLED = True

# Check for ROCm
IS_ROCM = torch.version.hip is not None

def clone_rocwmma():
    """Clone rocWMMA v2 from rocm-libraries repo to third_party directory."""
    rocm_libs_dir = os.path.join(THIRD_PARTY_DIR, "rocm-libraries")
    rocwmma_include = None
    
    if os.path.exists(rocm_libs_dir):
        print(f"rocm-libraries already exists at {rocm_libs_dir}")
        rocwmma_include = os.path.join(rocm_libs_dir, "projects", "rocwmma", "library", "include")
    else:
        print("Cloning rocWMMA v2 from rocm-libraries...")
        os.makedirs(THIRD_PARTY_DIR, exist_ok=True)
        
        # Use sparse checkout to only get rocwmma
        clone_cmds = [
            f'git clone --filter=blob:none --sparse https://github.com/ROCm/rocm-libraries.git "{rocm_libs_dir}"',
            f'cd "{rocm_libs_dir}" && git sparse-checkout set projects/rocwmma'
        ]
        
        for cmd in clone_cmds:
            ret = os.system(cmd)
            if ret != 0:
                print(f"Warning: Failed to execute: {cmd}")
                return None
        
        rocwmma_include = os.path.join(rocm_libs_dir, "projects", "rocwmma", "library", "include")
    
    if rocwmma_include and os.path.exists(rocwmma_include):
        print(f"rocWMMA v2 include path: {rocwmma_include}")
        return rocwmma_include
    else:
        print("Warning: rocWMMA include path not found")
        return None


def run_instantiations(src_dir: str):
    base_path = Path(src_dir)
    py_files = [
        path for path in base_path.rglob('*.py')
        if path.is_file()
    ]

    for py_file in py_files:
        print(f"Running: {py_file}")
        os.system(f"python {py_file}")


def get_instantiations(src_dir: str):
    # get all .cu files under src_dir
    base_path = Path(src_dir)
    return [
        os.path.join(src_dir, str(path.relative_to(base_path)))
        for path in base_path.rglob('*')
        if path.is_file() and path.suffix == ".cu"
    ]


# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"8.0", "8.6", "8.7", "8.9", "9.0"}

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0

ext_modules = []
cmdclass = {}

if IS_ROCM:
    print("Building for ROCm (AMD GPUs)")
    
    # Get ROCm architecture
    def get_rocm_arch():
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return props.gcnArchName.split(':')[0]
        except:
            pass
        rocm_arch = os.environ.get("ROCM_ARCH", None)
        if rocm_arch:
            return rocm_arch
        return "gfx1100"  # default
    
    rocm_arch = get_rocm_arch()
    print(f"Detected ROCm architecture: {rocm_arch}")
    
    # Clone rocWMMA v2
    rocwmma_include = clone_rocwmma()
    
    # Compiler flags for ROCm
    debug = os.environ.get("SA_DEBUG", "0") == "1"
    base_flags = ["-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={ABI}", "-DUSE_ROCM=1",
                  "-U__HIP_NO_HALF_CONVERSIONS__"]
    debug_flags = ["-O0", "-g3", "-ggdb", "-fno-inline", "-fno-omit-frame-pointer"] if debug else ["-O3"]
    
    rocm_hipcc = base_flags + debug_flags + [f"--offload-arch={rocm_arch}"]
    
    # Add architecture-specific defines for our kernel code
    if rocm_arch.startswith("gfx9"):
        rocm_hipcc.append("-DSA_ARCH_MI_SERIES=1")
    elif rocm_arch.startswith("gfx10") or rocm_arch.startswith("gfx11"):
        rocm_hipcc.append("-DSA_ARCH_RDNA_SERIES=1")
    
    # Windows-specific: avoid GPU RDC which causes linker issues
    if sys.platform == "win32":
        rocm_hipcc.append("-fno-gpu-rdc")
    
    rocm_hipcc.append(f"-D__ROCM_ARCH_{rocm_arch.upper()}")
    
    rocm_cxx = base_flags + debug_flags
    
    include_dirs = []
    if rocwmma_include:
        include_dirs.append(rocwmma_include)
        print(f"Using rocWMMA v2 from: {rocwmma_include}")
    else:
        print("Warning: Using system rocWMMA (may be older v1.x version)")
    
    is_mi_series = rocm_arch.startswith("gfx9")
    
    ext_kwargs = {
        "extra_compile_args": {"cxx": rocm_cxx, "nvcc": rocm_hipcc},
    }
    if include_dirs:
        ext_kwargs["include_dirs"] = include_dirs
    
    # Build qattn_rocm extension with FP16 kernels (all archs)
    qattn_sources = [
        "csrc/qattn/rocm/pybind_rocm.cpp",
        "csrc/qattn/rocm/sgattn_f16.cu",
        "csrc/qattn/rocm/launch_sgattn_f16.cu",
    ]
    
    if is_mi_series:
        qattn_sources.extend([
            "csrc/qattn/rocm/launch_sgattn.cu",
            "csrc/qattn/rocm/sgattn.cu"
        ])
        print(f"Building _qattn with FP8+FP16 for MI-series GPU ({rocm_arch})")
    else:
        print(f"Building _qattn with FP16 only for RDNA GPU ({rocm_arch})")
    
    ext_modules.append(
        CUDAExtension(
            "spas_sage_attn._qattn",
            sources=qattn_sources,
            **ext_kwargs
        )
    )
    
    # Build fused extension
    ext_modules.append(
        CUDAExtension(
            "spas_sage_attn._fused",
            sources=["csrc/fused/rocm/pybind_rocm.cpp", "csrc/fused/rocm/fused.cu"],
            **ext_kwargs
        )
    )
    
    cmdclass = {"build_ext": BuildExtension}

else:
    # NVIDIA CUDA build
    # Compiler flags.
    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
    NVCC_FLAGS = [
        "-O3",
        "-std=c++17",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--use_fast_math",
        "--threads=8",
        "-Xptxas=-v",
        "-diag-suppress=174", # suppress the specific warning
        "-Xcompiler", "-include,cassert", # fix error occurs when compiling for SM90+ with newer CUDA toolkits
    ]

    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

    if CUDA_HOME is None:
        raise RuntimeError(
            "Cannot find CUDA_HOME. CUDA must be available to build the package.")

    def get_nvcc_cuda_version(cuda_dir: str) -> Version:
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                              universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = parse(output[release_idx].split(",")[0])
        return nvcc_cuda_version

    def get_torch_arch_list() -> Set[str]:
        env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
        if env_arch_list is None:
            return set()

        torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
        if not torch_arch_list:
            return set()

        valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
        arch_list = torch_arch_list.intersection(valid_archs)
        if not arch_list:
            raise RuntimeError(
                "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
                f"variable ({env_arch_list}) is supported. "
                f"Supported CUDA architectures are: {valid_archs}.")
        invalid_arch_list = torch_arch_list - valid_archs
        if invalid_arch_list:
            warnings.warn(
                f"Unsupported CUDA architectures ({invalid_arch_list}) are "
                "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
                f"({env_arch_list}). Supported CUDA architectures are: "
                f"{valid_archs}.")
        return arch_list

    compute_capabilities = get_torch_arch_list()
    if not compute_capabilities:
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            major, minor = torch.cuda.get_device_capability(i)
            if major < 8:
                raise RuntimeError(
                    "GPUs with compute capability below 8.0 are not supported.")
            compute_capabilities.add(f"{major}.{minor}")

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    if not compute_capabilities:
        raise RuntimeError("No GPUs found. Please specify the target GPU architectures or build on a machine with GPUs.")

    if nvcc_cuda_version < Version("12.0"):
        raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
    if nvcc_cuda_version < Version("12.4"):
        if any(cc.startswith("8.9") for cc in compute_capabilities):
            raise RuntimeError(
                "CUDA 12.4 or higher is required for compute capability 8.9.")
        if any(cc.startswith("9.0") for cc in compute_capabilities):
            raise RuntimeError(
                "CUDA 12.4 or higher is required for compute capability 9.0.")
    if nvcc_cuda_version < Version("12.8"):
        warnings.warn("CUDA 12.8 or higher is required for Sage2++")
        SAGE2PP_ENABLED = False

    for capability in compute_capabilities:
        num = capability.replace(".", "")
        if num == '90':
            num = '90a'
            HAS_SM90 = True
            CXX_FLAGS += ["-DHAS_SM90"]
        if num == '80' or num == '86' or num == '87':
            SAGE2PP_ENABLED = False
        
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

    if SAGE2PP_ENABLED:
        CXX_FLAGS += ["-DSAGE2PP_ENABLED"]

    run_instantiations("csrc/qattn/instantiations_sm80")
    run_instantiations("csrc/qattn/instantiations_sm89")
    run_instantiations("csrc/qattn/instantiations_sm90")

    sources = [
        "csrc/qattn/pybind.cpp",
        "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
        "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
    ] + get_instantiations("csrc/qattn/instantiations_sm80") + get_instantiations("csrc/qattn/instantiations_sm89")

    if HAS_SM90:
        sources += ["csrc/qattn/qk_int_sv_f8_cuda_sm90.cu", ]
        sources += get_instantiations("csrc/qattn/instantiations_sm90")

    qattn_extension = CUDAExtension(
        name="spas_sage_attn._qattn",
        sources=sources,
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
        extra_link_args=['-lcuda'],
    )
    ext_modules.append(qattn_extension)

    fused_extension = CUDAExtension(
        name="spas_sage_attn._fused",
        sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(fused_extension)

    cmdclass = {"build_ext": BuildExtension}

setup(
    name='spas_sage_attn', 
    version='0.1.0',  
    author='Jintao Zhang, Chendong Xiang, Haofeng Huang',  
    author_email='jt-zhang6@gmail.com', 
    packages=find_packages(),  
    description='Accurate and efficient Sparse SageAttention.',  
    long_description=open('README.md', encoding='utf-8').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/thu-ml/SpargeAttn', 
    license='BSD 3-Clause License', 
    python_requires='>=3.9', 
    classifiers=[  
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers',  
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
